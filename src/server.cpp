// openai-compatible tts server for qwen3-tts.cpp
//
// endpoints:
//   GET  /health              - health check
//   GET  /v1/models           - list loaded model
//   GET  /v1/audio/voices     - list available voices
//   POST /v1/audio/voices     - create custom voice from reference audio
//   DELETE /v1/audio/voices/X - delete custom voice
//   POST /v1/audio/speech     - synthesize speech (supports voice cloning)

#include "qwen3_tts.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;
using namespace qwen3_tts;

// language string to model language_id
static int language_to_id(const std::string & lang) {
    if (lang.empty() || lang == "en") return 2050;
    if (lang == "ru") return 2069;
    if (lang == "zh") return 2055;
    if (lang == "ja") return 2058;
    if (lang == "ko") return 2064;
    if (lang == "de") return 2053;
    if (lang == "fr") return 2061;
    if (lang == "es") return 2054;
    if (lang == "it") return 2070;
    if (lang == "pt") return 2071;
    return 2050;
}

// encode float32 audio samples as a WAV byte buffer (16-bit PCM)
static std::string encode_wav(const std::vector<float> & samples, int sample_rate) {
    const int num_channels = 1;
    const int bits_per_sample = 16;
    const int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const int block_align = num_channels * bits_per_sample / 8;
    const int data_size = (int)samples.size() * block_align;
    const int file_size = 36 + data_size;

    std::string buf;
    buf.resize(44 + data_size);
    char * p = buf.data();

    auto write_u32 = [](char * dst, uint32_t v) {
        dst[0] = (char)(v & 0xff);
        dst[1] = (char)((v >> 8) & 0xff);
        dst[2] = (char)((v >> 16) & 0xff);
        dst[3] = (char)((v >> 24) & 0xff);
    };
    auto write_u16 = [](char * dst, uint16_t v) {
        dst[0] = (char)(v & 0xff);
        dst[1] = (char)((v >> 8) & 0xff);
    };

    // RIFF header
    memcpy(p, "RIFF", 4);      write_u32(p + 4, file_size);
    memcpy(p + 8, "WAVE", 4);

    // fmt chunk
    memcpy(p + 12, "fmt ", 4);  write_u32(p + 16, 16);
    write_u16(p + 20, 1);       // PCM
    write_u16(p + 22, num_channels);
    write_u32(p + 24, sample_rate);
    write_u32(p + 28, byte_rate);
    write_u16(p + 32, block_align);
    write_u16(p + 34, bits_per_sample);

    // data chunk
    memcpy(p + 36, "data", 4);  write_u32(p + 40, data_size);

    // convert float32 [-1,1] to int16
    int16_t * dst = reinterpret_cast<int16_t *>(p + 44);
    for (size_t i = 0; i < samples.size(); i++) {
        float s = samples[i];
        if (s > 1.0f)  s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        dst[i] = (int16_t)(s * 32767.0f);
    }

    return buf;
}

// encode float32 audio samples as raw PCM (int16, little-endian)
static std::string encode_pcm(const std::vector<float> & samples) {
    std::string buf;
    buf.resize(samples.size() * sizeof(int16_t));
    int16_t * dst = reinterpret_cast<int16_t *>(buf.data());
    for (size_t i = 0; i < samples.size(); i++) {
        float s = samples[i];
        if (s > 1.0f)  s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        dst[i] = (int16_t)(s * 32767.0f);
    }
    return buf;
}

// in-memory custom voice store
struct custom_voice {
    std::string name;
    std::vector<float> embedding; // 1024-dim speaker embedding
    // ICL voice cloning data (optional)
    std::string ref_text;
    std::vector<int32_t> ref_codes;
    int32_t n_ref_frames = 0;
};

struct server_params {
    std::string model;
    std::string vocoder;
    std::string hf_repo;     // e.g. "khimaros/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF:Q8_0"
    std::string hf_file;     // override filename within --hf-repo
    std::string hf_repo_v;   // vocoder HF repo
    std::string hf_file_v;   // override filename within --hf-repo-v
    std::string host      = "127.0.0.1";
    int         port      = 8080;
    int         n_threads = 4;
    bool        verbose   = false;
    float       temperature        = 0.9f;
    int         top_k              = 50;
    float       repetition_penalty = 1.05f;
    int64_t     seed               = -1;
};

// download a file from a huggingface repo, returns local cache path
static std::string hf_download(const std::string & repo, const std::string & filename) {
    std::string cmd = "hf download \"" + repo + "\" \"" + filename + "\" --quiet";
    FILE * fp = popen(cmd.c_str(), "r");
    if (!fp) return "";

    std::string path;
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp)) {
        path = buf;
    }
    int status = pclose(fp);
    if (status != 0) return "";

    // trim trailing whitespace
    while (!path.empty() && (path.back() == '\n' || path.back() == '\r' || path.back() == ' ')) {
        path.pop_back();
    }
    return path;
}

// resolve "user/RepoName-GGUF:QUANT" to a GGUF filename and download it.
// if file_override is set, use that filename instead of deriving one.
// default_quant is used when no :QUANT suffix is present.
static std::string hf_resolve(const std::string & repo_spec, const std::string & file_override,
                               const std::string & default_quant = "Q8_0") {
    std::string repo = repo_spec;
    std::string quant = default_quant;
    auto colon = repo.rfind(':');
    if (colon != std::string::npos) {
        quant = repo.substr(colon + 1);
        repo = repo.substr(0, colon);
    }

    std::vector<std::string> candidates;
    if (!file_override.empty()) {
        candidates.push_back(file_override);
    } else {
        // derive filename: strip "-GGUF" suffix (case-insensitive), try both quant cases
        std::string basename = repo;
        auto slash = basename.rfind('/');
        if (slash != std::string::npos) basename = basename.substr(slash + 1);
        if (basename.size() > 5) {
            std::string tail = basename.substr(basename.size() - 5);
            for (char & c : tail) c = (char)std::toupper((unsigned char)c);
            if (tail == "-GGUF") basename = basename.substr(0, basename.size() - 5);
        }
        candidates.push_back(basename + "-" + quant + ".gguf");
        std::string lquant = quant;
        for (char & c : lquant) c = (char)std::tolower((unsigned char)c);
        if (lquant != quant) candidates.push_back(basename + "-" + lquant + ".gguf");
    }

    for (const auto & gguf_file : candidates) {
        fprintf(stderr, "downloading %s/%s ...\n", repo.c_str(), gguf_file.c_str());
        std::string local_path = hf_download(repo, gguf_file);
        if (!local_path.empty()) return local_path;
    }
    fprintf(stderr, "fatal: failed to download from %s (tried:", repo.c_str());
    for (const auto & c : candidates) fprintf(stderr, " %s", c.c_str());
    fprintf(stderr, ")\n");
    return "";
}

static void print_usage(const char * program) {
    fprintf(stderr, "usage: %s [options] (-m <model.gguf> | -hf <repo:quant>)\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -m,  --model <file>             TTS model GGUF file\n");
    fprintf(stderr, "  -v,  --vocoder <file>           vocoder GGUF file (default: same dir as model)\n");
    fprintf(stderr, "  -hf, --hf-repo <repo[:quant]>   HuggingFace model repo (default quant: Q8_0)\n");
    fprintf(stderr, "       --hf-file <file>            override GGUF filename within --hf-repo\n");
    fprintf(stderr, "       --hf-repo-v <repo[:quant]>  HuggingFace vocoder repo\n");
    fprintf(stderr, "       --hf-file-v <file>          override GGUF filename within --hf-repo-v\n");
    fprintf(stderr, "  -H,  --host <host>              listen host (default: 127.0.0.1)\n");
    fprintf(stderr, "  -p,  --port <port>              listen port (default: 8080)\n");
    fprintf(stderr, "  -j,  --threads <n>              compute threads (default: 4)\n");
    fprintf(stderr, "  -V,  --verbose                  print per-stage progress and timing\n");
    fprintf(stderr, "       --temperature <f>           sampling temperature default (default: 0.9)\n");
    fprintf(stderr, "       --top-k <n>                 top-k sampling default (default: 50)\n");
    fprintf(stderr, "       --repetition-penalty <f>    repetition penalty default (default: 1.05)\n");
    fprintf(stderr, "       --seed <n>                  default sampling seed (default: -1 = random)\n");
    fprintf(stderr, "  -h,  --help                     show this help\n");
}

static bool parse_args(int argc, char ** argv, server_params & sp) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "error: missing model path\n"); return false; }
            sp.model = argv[i];
        } else if (arg == "-v" || arg == "--vocoder") {
            if (++i >= argc) { fprintf(stderr, "error: missing vocoder path\n"); return false; }
            sp.vocoder = argv[i];
        } else if (arg == "-H" || arg == "--host") {
            if (++i >= argc) { fprintf(stderr, "error: missing host\n"); return false; }
            sp.host = argv[i];
        } else if (arg == "-p" || arg == "--port") {
            if (++i >= argc) { fprintf(stderr, "error: missing port\n"); return false; }
            sp.port = std::stoi(argv[i]);
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) { fprintf(stderr, "error: missing threads\n"); return false; }
            sp.n_threads = std::stoi(argv[i]);
        } else if (arg == "-V" || arg == "--verbose") {
            sp.verbose = true;
        } else if (arg == "-hf" || arg == "--hf-repo") {
            if (++i >= argc) { fprintf(stderr, "error: missing hf repo\n"); return false; }
            sp.hf_repo = argv[i];
        } else if (arg == "--hf-file") {
            if (++i >= argc) { fprintf(stderr, "error: missing hf file\n"); return false; }
            sp.hf_file = argv[i];
        } else if (arg == "--hf-repo-v") {
            if (++i >= argc) { fprintf(stderr, "error: missing hf vocoder repo\n"); return false; }
            sp.hf_repo_v = argv[i];
        } else if (arg == "--hf-file-v") {
            if (++i >= argc) { fprintf(stderr, "error: missing hf vocoder file\n"); return false; }
            sp.hf_file_v = argv[i];
        } else if (arg == "--temperature") {
            if (++i >= argc) { fprintf(stderr, "error: missing temperature\n"); return false; }
            sp.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) { fprintf(stderr, "error: missing top-k\n"); return false; }
            sp.top_k = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) { fprintf(stderr, "error: missing repetition-penalty\n"); return false; }
            sp.repetition_penalty = std::stof(argv[i]);
        } else if (arg == "--seed") {
            if (++i >= argc) { fprintf(stderr, "error: missing seed\n"); return false; }
            sp.seed = std::stoll(argv[i]);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    if (sp.model.empty() && sp.hf_repo.empty()) {
        fprintf(stderr, "error: -m <model> or --hf-repo <repo> is required\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    server_params sp;
    if (!parse_args(argc, argv, sp)) {
        print_usage(argv[0]);
        return 1;
    }

    // resolve --hf-repo to local file paths
    if (!sp.hf_repo.empty()) {
        sp.model = hf_resolve(sp.hf_repo, sp.hf_file);
        if (sp.model.empty()) return 1;
        fprintf(stderr, "resolved model: %s\n", sp.model.c_str());
    }
    if (!sp.hf_repo_v.empty()) {
        sp.vocoder = hf_resolve(sp.hf_repo_v, sp.hf_file_v, "F16");
        if (sp.vocoder.empty()) return 1;
        fprintf(stderr, "resolved vocoder: %s\n", sp.vocoder.c_str());
    }

    // load models
    Qwen3TTS tts;
    fprintf(stderr, "loading model: %s\n", sp.model.c_str());
    if (!sp.vocoder.empty()) {
        fprintf(stderr, "loading vocoder: %s\n", sp.vocoder.c_str());
    }
    if (!tts.load_model_files(sp.model, sp.vocoder)) {
        fprintf(stderr, "fatal: %s\n", tts.get_error().c_str());
        return 1;
    }
    fprintf(stderr, "models loaded (type=%s, speakers=%zu)\n",
            tts.get_model_type().c_str(), tts.get_speaker_names().size());

    // derive model id from filename (e.g. "qwen3-tts-0.6b-f16" from path)
    std::string model_id = sp.model;
    auto slash = model_id.rfind('/');
    if (slash != std::string::npos) model_id = model_id.substr(slash + 1);
    auto dot = model_id.rfind('.');
    if (dot != std::string::npos) model_id = model_id.substr(0, dot);

    // synthesis is not thread-safe, serialize all requests
    std::mutex synth_mutex;

    // custom voice store (voice_id -> voice data)
    std::map<std::string, custom_voice> voices;
    std::mutex voices_mutex;
    int next_voice_id = 1;

    httplib::Server svr;

    // log all requests
    svr.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        fprintf(stderr, "%s %s%s%s -> %d\n",
                req.method.c_str(), req.path.c_str(),
                req.params.empty() ? "" : "?",
                req.params.empty() ? "" : [&]() {
                    static thread_local std::string qs;
                    qs.clear();
                    for (auto & [k, v] : req.params) {
                        if (!qs.empty()) qs += '&';
                        qs += k + "=" + v;
                    }
                    return qs.c_str();
                }(),
                res.status);
    });

    // --- GET /health ---
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // --- GET /v1/models ---
    svr.Get("/v1/models", [&model_id](const httplib::Request &, httplib::Response & res) {
        json models = {
            {"object", "list"},
            {"data", json::array({
                {{"id", model_id}, {"object", "model"}, {"owned_by", "qwen"}},
            })},
        };
        res.set_content(models.dump(), "application/json");
    });

    // --- GET /v1/audio/voices ---
    svr.Get("/v1/audio/voices", [&model_id, &tts, &voices, &voices_mutex](const httplib::Request &, httplib::Response & res) {
        json voice_list = json::array({"default"});

        // add built-in speakers from model metadata (custom_voice models)
        for (auto & name : tts.get_speaker_names()) {
            voice_list.push_back(name);
        }

        // add user-created cloned voices
        {
            std::lock_guard<std::mutex> lock(voices_mutex);
            for (auto & [id, v] : voices) {
                voice_list.push_back(id);
            }
        }
        res.set_content(json({{model_id, voice_list}}).dump(), "application/json");
    });

    // --- POST /v1/audio/voices --- create custom voice from reference audio
    svr.Post("/v1/audio/voices",
        [&tts, &synth_mutex, &voices, &voices_mutex, &next_voice_id](const httplib::Request & req, httplib::Response & res) {

        // runtime audio cloning needs the speaker encoder, which only ships in the Base variant
        if (!tts.has_speaker_encoder()) {
            res.status = 400;
            json err = {{"error", {
                {"message", "this model variant (" + tts.get_model_type() +
                            ") does not support voice cloning from audio; "
                            "use the Base variant, or pick a built-in voice via GET /v1/audio/voices"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // expect multipart form: name (string) + audio_sample (file)
        if (!req.has_file("audio_sample")) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'audio_sample' file is required","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }
        std::string name = "custom";
        if (req.has_param("name")) name = req.get_param_value("name");
        if (req.has_file("name")) name = req.get_file_value("name").content;

        auto audio_file = req.get_file_value("audio_sample");

        // write to temp file for the encoder (expects a file path)
        char tmppath[] = "/tmp/qwen3tts_voice_XXXXXX.wav";
        int fd = mkstemps(tmppath, 4);
        if (fd < 0) {
            res.status = 500;
            res.set_content(R"({"error":{"message":"failed to create temp file","type":"server_error"}})",
                            "application/json");
            return;
        }
        write(fd, audio_file.content.data(), audio_file.content.size());
        close(fd);

        // optional ref_text for ICL voice cloning
        std::string ref_text;
        if (req.has_param("ref_text")) ref_text = req.get_param_value("ref_text");
        if (req.has_file("ref_text")) ref_text = req.get_file_value("ref_text").content;

        // extract speaker embedding (optional when ref_text is provided for ICL mode)
        std::vector<float> embedding;
        bool ok;
        {
            std::lock_guard<std::mutex> lock(synth_mutex);
            ok = tts.extract_speaker_embedding(tmppath, embedding);
        }

        if (!ok && ref_text.empty()) {
            unlink(tmppath);
            res.status = 400;
            json err = {{"error", {
                {"message", "failed to extract speaker embedding: " + tts.get_error()},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // ICL mode: also encode reference audio to discrete speech codes
        std::vector<int32_t> ref_codes;
        int32_t n_ref_frames = 0;
        if (!ref_text.empty()) {
            std::vector<float> samples;
            int sample_rate = 0;
            if (!qwen3_tts::load_audio_file(tmppath, samples, sample_rate)) {
                unlink(tmppath);
                res.status = 400;
                res.set_content(R"({"error":{"message":"failed to load audio for codec encoding","type":"invalid_request_error"}})",
                                "application/json");
                return;
            }

            // resample to 24kHz if needed
            if (sample_rate != 24000 && sample_rate > 0) {
                int64_t new_len = (int64_t)samples.size() * 24000 / sample_rate;
                std::vector<float> resampled(new_len);
                for (int64_t i = 0; i < new_len; i++) {
                    float src = (float)i * sample_rate / 24000.0f;
                    int idx = (int)src;
                    float frac = src - idx;
                    if (idx + 1 < (int)samples.size()) {
                        resampled[i] = samples[idx] * (1 - frac) + samples[idx + 1] * frac;
                    } else {
                        resampled[i] = samples[std::min(idx, (int)samples.size() - 1)];
                    }
                }
                samples = std::move(resampled);
            }

            bool codec_ok;
            {
                std::lock_guard<std::mutex> lock(synth_mutex);
                codec_ok = tts.encode_speech_codes(samples.data(),
                                                    (int32_t)samples.size(),
                                                    ref_codes, n_ref_frames);
            }
            if (!codec_ok) {
                unlink(tmppath);
                res.status = 500;
                json err = {{"error", {
                    {"message", "failed to encode speech codes: " + tts.get_error()},
                    {"type", "server_error"},
                }}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            fprintf(stderr, "encoded %d reference frames for ICL voice cloning\n", n_ref_frames);
        }

        unlink(tmppath);

        // store voice
        std::string voice_id;
        {
            std::lock_guard<std::mutex> lock(voices_mutex);
            voice_id = "voice_" + std::to_string(next_voice_id++);
            voices[voice_id] = {name, std::move(embedding), ref_text,
                                std::move(ref_codes), n_ref_frames};
        }

        fprintf(stderr, "created voice '%s' (id: %s%s)\n", name.c_str(), voice_id.c_str(),
                ref_text.empty() ? "" : ", ICL mode");
        json resp = {{"id", voice_id}, {"name", name}};
        if (!ref_text.empty()) {
            resp["mode"] = "icl";
            resp["ref_frames"] = n_ref_frames;
        }
        res.set_content(resp.dump(), "application/json");
    });

    // --- DELETE /v1/audio/voices/:id ---
    svr.Delete(R"(/v1/audio/voices/(.+))",
        [&voices, &voices_mutex](const httplib::Request & req, httplib::Response & res) {
        std::string voice_id = req.matches[1];
        std::lock_guard<std::mutex> lock(voices_mutex);
        if (voices.erase(voice_id)) {
            res.set_content(R"({"deleted":true})", "application/json");
        } else {
            res.status = 404;
            res.set_content(R"({"error":{"message":"voice not found","type":"not_found"}})",
                            "application/json");
        }
    });

    // --- POST /v1/audio/speech ---
    svr.Post("/v1/audio/speech",
        [&tts, &synth_mutex, &sp, &voices, &voices_mutex](const httplib::Request & req, httplib::Response & res) {

        // parse request body
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception &) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"invalid JSON","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        // extract parameters
        std::string input = body.value("input", "");
        if (input.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'input' is required","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        // openai text limit is 4096 chars
        if (input.size() > 4096) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'input' exceeds 4096 characters","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        std::string response_format = body.value("response_format", "wav");
        std::string voice           = body.value("voice", "");
        std::string instructions    = body.value("instructions", "");
        std::string language        = body.value("language", "en");
        float       temperature     = body.value("temperature", sp.temperature);
        int         top_k           = body.value("top_k", sp.top_k);
        float       repetition_penalty = body.value("repetition_penalty", sp.repetition_penalty);
        int64_t     seed               = body.value("seed", sp.seed);

        fprintf(stderr, "request: voice=%s lang=%s fmt=%s temp=%.2f seed=%lld len=%zu\n",
                voice.empty() ? "default" : voice.c_str(),
                language.c_str(), response_format.c_str(),
                temperature, (long long)seed, input.size());

        // validate response format
        if (response_format != "wav" && response_format != "pcm") {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported response_format '" + response_format +
                            "', supported: wav, pcm"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // resolve voice to speaker embedding (and optional ICL data)
        std::vector<float> voice_embedding;
        std::string voice_ref_text;
        std::vector<int32_t> voice_ref_codes;
        int32_t voice_n_ref_frames = 0;
        if (!voice.empty() && voice != "default") {
            // try built-in speaker first (custom_voice models)
            if (tts.get_speaker_id(voice) >= 0) {
                std::lock_guard<std::mutex> lock(synth_mutex);
                if (!tts.get_speaker_embedding(voice, voice_embedding)) {
                    res.status = 500;
                    json err = {{"error", {
                        {"message", "failed to get speaker embedding: " + tts.get_error()},
                        {"type", "server_error"},
                    }}};
                    res.set_content(err.dump(), "application/json");
                    return;
                }
            } else {
                // try user-created cloned voice
                std::lock_guard<std::mutex> lock(voices_mutex);
                auto it = voices.find(voice);
                if (it == voices.end()) {
                    res.status = 400;
                    json err = {{"error", {
                        {"message", "unknown voice '" + voice + "'"},
                        {"type", "invalid_request_error"},
                    }}};
                    res.set_content(err.dump(), "application/json");
                    return;
                }
                voice_embedding = it->second.embedding;
                voice_ref_text = it->second.ref_text;
                voice_ref_codes = it->second.ref_codes;
                voice_n_ref_frames = it->second.n_ref_frames;
            }
        }

        // set up synthesis params
        tts_params params;
        params.n_threads          = sp.n_threads;
        params.temperature        = temperature;
        params.top_k              = top_k;
        params.repetition_penalty = repetition_penalty;
        params.seed               = seed;
        params.language_id        = language_to_id(language);
        params.print_progress     = sp.verbose;
        params.print_timing       = sp.verbose;
        params.instructions       = instructions;
        params.ref_text           = voice_ref_text;

        // synthesize (serialized), using voice embedding if provided
        tts_result result;
        {
            std::lock_guard<std::mutex> lock(synth_mutex);
            if (!voice_ref_codes.empty()) {
                result = tts.synthesize_with_embedding(
                    input, voice_embedding.data(), (int32_t)voice_embedding.size(), params,
                    voice_ref_codes.data(), voice_n_ref_frames);
            } else if (!voice_embedding.empty()) {
                result = tts.synthesize_with_embedding(
                    input, voice_embedding.data(), (int32_t)voice_embedding.size(), params);
            } else {
                result = tts.synthesize(input, params);
            }
        }

        if (!result.success) {
            res.status = 500;
            json err = {{"error", {
                {"message", "synthesis failed: " + result.error_msg},
                {"type", "server_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        if (result.audio.empty()) {
            res.status = 500;
            res.set_content(R"({"error":{"message":"synthesis produced no audio","type":"server_error"}})",
                            "application/json");
            return;
        }

        fprintf(stderr, "synthesized %.2fs audio (%zu samples) in %lldms\n",
                (float)result.audio.size() / result.sample_rate,
                result.audio.size(), (long long)result.t_total_ms);

        // encode and return audio
        if (response_format == "pcm") {
            std::string audio_data = encode_pcm(result.audio);
            res.set_content(std::move(audio_data), "audio/pcm");
        } else {
            std::string audio_data = encode_wav(result.audio, result.sample_rate);
            res.set_content(std::move(audio_data), "audio/wav");
        }
    });

    fprintf(stderr, "server listening on %s:%d\n", sp.host.c_str(), sp.port);
    if (!svr.listen(sp.host, sp.port)) {
        fprintf(stderr, "fatal: failed to bind to %s:%d\n", sp.host.c_str(), sp.port);
        return 1;
    }

    return 0;
}
