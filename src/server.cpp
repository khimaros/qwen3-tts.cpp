// openai-compatible tts server for qwen3-tts.cpp
//
// endpoints:
//   GET  /health              - health check
//   GET  /v1/models           - list loaded model
//   GET  /v1/audio/languages  - list supported languages
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

// supported languages and their model codec token IDs
static const std::vector<std::pair<std::string, int>> SUPPORTED_LANGUAGES = {
    {"en", 2050}, {"zh", 2055}, {"ja", 2058}, {"ko", 2064}, {"ru", 2069},
    {"de", 2053}, {"fr", 2061}, {"es", 2054}, {"it", 2070}, {"pt", 2071},
};

// language string to model language_id (returns -1 if unknown)
static int language_to_id(const std::string & lang) {
    if (lang.empty()) return 2050;
    for (const auto & [code, id] : SUPPORTED_LANGUAGES) {
        if (lang == code) return id;
    }
    return -1;
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

// emit a 44-byte WAV header with placeholder sizes for streaming.
// clients that tolerate non-finite RIFF/data sizes (ffmpeg, vlc, most players)
// can start playing before the full body arrives.
static std::string wav_streaming_header(int sample_rate) {
    const int num_channels = 1;
    const int bits_per_sample = 16;
    const int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const int block_align = num_channels * bits_per_sample / 8;

    std::string buf;
    buf.resize(44);
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

    memcpy(p, "RIFF", 4);       write_u32(p + 4, 0xFFFFFFFF);
    memcpy(p + 8, "WAVE", 4);
    memcpy(p + 12, "fmt ", 4);  write_u32(p + 16, 16);
    write_u16(p + 20, 1);
    write_u16(p + 22, num_channels);
    write_u32(p + 24, sample_rate);
    write_u32(p + 28, byte_rate);
    write_u16(p + 32, block_align);
    write_u16(p + 34, bits_per_sample);
    memcpy(p + 36, "data", 4);  write_u32(p + 40, 0xFFFFFFFF);
    return buf;
}

// http content type for a response_format
static const char * content_type_for(const std::string & fmt) {
    if (fmt == "mp3")  return "audio/mpeg";
    if (fmt == "opus") return "audio/ogg";
    if (fmt == "pcm")  return "audio/pcm";
    return "audio/wav";
}

// encode a full result into a self-contained byte buffer for the chosen format
static std::string encode_format(const std::string & fmt,
                                 const std::vector<float> & samples, int sample_rate) {
    audio_codec codec;
    if (codec_from_name(fmt, codec)) return encode_compressed(codec, samples, sample_rate);
    if (fmt == "pcm") return encode_pcm(samples);
    return encode_wav(samples, sample_rate);
}

// minimal RFC 4648 base64 encoder (no line wrapping)
static std::string base64_encode(const char * data, size_t len) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    int val = 0, valb = -6;
    for (size_t i = 0; i < len; i++) {
        val = (val << 8) + (uint8_t)data[i];
        valb += 8;
        while (valb >= 0) {
            out.push_back(tbl[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(tbl[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

// build the speech.audio.done SSE payload. usage mirrors openai (counts user
// content tokens only); timings mirrors llama.cpp's prompt/predicted schema
// for llama-swap compatibility. The tts-specific stages (speaker encoder,
// vocoder, text tokenizer) are surfaced as extra keys — llama-swap ignores
// unknown fields, and our own clients can render the full breakdown.
//
// Semantics:
//   usage.input_tokens   — tokens in the user's `input` text (maps to openai billing).
//   usage.output_tokens  — generated audio codec frames.
//   timings.prompt_n     — real transformer prefill length (text + instruct + ref_text
//                          + ref_codes + framing), i.e. work done before the first
//                          generated audio token.
//   timings.prompt_ms    — build_prefill_graph + forward_prefill wall time.
//   timings.predicted_n  — n_audio_tokens.
//   timings.predicted_ms — transformer autoregressive loop only (excludes vocoder and
//                          prefill), so predicted_per_second reflects pure transformer
//                          throughput comparable to llama-server.
static std::string build_done_event(const tts_result & result) {
    const int32_t input_tokens   = result.n_text_tokens;
    const int32_t output_tokens  = result.n_audio_tokens;
    const int32_t prefill_tokens = result.n_prefill_tokens;
    const int64_t prompt_ms      = result.t_prefill_ms;

    // transformer decode loop only. if get_last_prefill_ms() overshoots
    // t_generate_ms by rounding, clamp to 0 rather than emit a negative.
    int64_t predicted_ms = result.t_generate_ms - result.t_prefill_ms;
    if (predicted_ms < 0) predicted_ms = 0;

    const double pps = prompt_ms    > 0 ? (double)prefill_tokens * 1000.0 / (double)prompt_ms    : 0.0;
    const double tps = predicted_ms > 0 ? (double)output_tokens  * 1000.0 / (double)predicted_ms : 0.0;

    json ev = {
        {"type", "speech.audio.done"},
        {"usage", {
            {"input_tokens",  input_tokens},
            {"output_tokens", output_tokens},
            {"total_tokens",  input_tokens + output_tokens},
        }},
        {"timings", {
            {"prompt_n",             prefill_tokens},
            {"predicted_n",          output_tokens},
            {"prompt_ms",            prompt_ms},
            {"predicted_ms",         predicted_ms},
            {"prompt_per_second",    pps},
            {"predicted_per_second", tps},
            // project extras (llama-swap ignores unknown keys):
            {"tokenize_ms",          result.t_tokenize_ms},
            {"encode_ms",            result.t_encode_ms},
            {"generate_ms",          result.t_generate_ms},
            {"decode_ms",            result.t_decode_ms},
            {"total_ms",             result.t_total_ms},
            {"n_text_tokens",        input_tokens},
        }},
    };
    return ev.dump();
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

    // CORS: allow cross-origin browser requests. OPTIONS preflight is handled
    // here; other methods fall through to normal routing with CORS headers set.
    svr.set_pre_routing_handler([](const httplib::Request & req, httplib::Response & res) {
        auto origin = req.get_header_value("Origin");
        if (!origin.empty()) {
            res.set_header("Access-Control-Allow-Origin", origin);
            res.set_header("Vary", "Origin");
        }
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept");
        res.set_header("Access-Control-Max-Age", "86400");

        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

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

    // --- GET /v1/audio/languages ---
    svr.Get("/v1/audio/languages", [](const httplib::Request &, httplib::Response & res) {
        json lang_list = json::array();
        for (const auto & [code, id] : SUPPORTED_LANGUAGES) {
            lang_list.push_back({{"code", code}, {"id", id}});
        }
        res.set_content(json({{"languages", lang_list}}).dump(), "application/json");
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
        std::string stream_format   = body.value("stream_format", "");
        std::string voice           = body.value("voice", "");
        std::string instructions    = body.value("instructions", "");
        std::string language        = body.value("language", "en");
        float       temperature     = body.value("temperature", sp.temperature);
        int         top_k           = body.value("top_k", sp.top_k);
        float       repetition_penalty = body.value("repetition_penalty", sp.repetition_penalty);
        int64_t     seed               = body.value("seed", sp.seed);
        int         stream_batch_size  = body.value("stream_batch_size", 0);
        if (stream_batch_size < 0) stream_batch_size = 0;
        if (stream_batch_size > 256) stream_batch_size = 256;

        fprintf(stderr, "request: voice=%s lang=%s fmt=%s temp=%.2f seed=%lld len=%zu\n",
                voice.empty() ? "default" : voice.c_str(),
                language.c_str(), response_format.c_str(),
                temperature, (long long)seed, input.size());

        // validate language
        int language_id = language_to_id(language);
        if (language_id < 0) {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported language '" + language +
                            "', see GET /v1/audio/languages"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // validate response format
        if (response_format != "wav" && response_format != "pcm" &&
            response_format != "mp3" && response_format != "opus") {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported response_format '" + response_format +
                            "', supported: wav, pcm, mp3, opus"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }
        audio_codec response_codec;
        if (codec_from_name(response_format, response_codec) && !compressed_audio_supported()) {
            res.status = 400;
            json err = {{"error", {
                {"message", "response_format '" + response_format + "' is not available: "
                            "server built without libav/ffmpeg"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // validate stream format (empty = one-shot, openai-spec values = chunked)
        if (!stream_format.empty() && stream_format != "audio" && stream_format != "sse") {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported stream_format '" + stream_format +
                            "', supported: audio, sse"},
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
        params.language_id        = language_id;
        params.print_progress     = sp.verbose;
        params.print_timing       = sp.verbose;
        params.instructions       = instructions;
        params.ref_text           = voice_ref_text;

        // live streaming path: when stream_format is set and stream_batch_size
        // > 0, synthesis runs INSIDE set_chunked_content_provider so PCM
        // batches flush to the wire as they're produced. stream_batch_size=0
        // preserves the legacy "synthesize-then-chunk" behavior for clients
        // that want a single delta event.
        const bool live_stream = !stream_format.empty() && stream_batch_size > 0;
        if (live_stream) {
            const bool is_sse = (stream_format == "sse");
            const bool is_wav = (response_format == "wav");
            audio_codec stream_codec;
            const bool is_compressed = codec_from_name(response_format, stream_codec);
            const char * ctype = is_sse ? "text/event-stream"
                                        : content_type_for(response_format);

            // capture synthesis inputs; move into provider lambda below.
            res.set_chunked_content_provider(ctype,
                [this_tts = &tts, input = std::move(input), params = std::move(params),
                 voice_embedding = std::move(voice_embedding),
                 voice_ref_codes = std::move(voice_ref_codes),
                 voice_n_ref_frames,
                 stream_batch_size, is_sse, is_wav, is_compressed, stream_codec,
                 synth_mutex = &synth_mutex, sample_rate_fallback = 24000]
                (size_t /*offset*/, httplib::DataSink & sink) mutable -> bool {
                    std::lock_guard<std::mutex> lock(*synth_mutex);

                    // wav header up front (audio mode only). for SSE, the per-delta
                    // bytes are raw pcm/mp3/opus — clients reconstruct the container.
                    bool header_written = false;
                    auto ensure_header = [&]() {
                        if (!header_written && !is_sse && is_wav) {
                            std::string hdr = wav_streaming_header(sample_rate_fallback);
                            sink.write(hdr.data(), hdr.size());
                        }
                        header_written = true;
                    };

                    // write a chunk of already-encoded bytes, as an sse delta or
                    // raw to the wire; empty chunks are skipped (muxer buffers).
                    auto emit = [&](const std::string & bytes) -> bool {
                        if (bytes.empty()) return true;
                        if (is_sse) {
                            json delta = {
                                {"type", "speech.audio.delta"},
                                {"audio", base64_encode(bytes.data(), bytes.size())},
                            };
                            std::string frame = "event: speech.audio.delta\ndata: "
                                              + delta.dump() + "\n\n";
                            return sink.write(frame.data(), frame.size());
                        }
                        return sink.write(bytes.data(), bytes.size());
                    };

                    compressed_encoder * enc = is_compressed
                        ? compressed_encoder_open(stream_codec, sample_rate_fallback) : nullptr;

                    streaming_opts sopts;
                    sopts.batch_size = stream_batch_size;
                    sopts.on_pcm = [&](const float * pcm, size_t n) -> bool {
                        if (enc) return emit(compressed_encoder_write(enc, pcm, n));
                        ensure_header();
                        return emit(encode_pcm(std::vector<float>(pcm, pcm + n)));
                    };

                    tts_result result;
                    if (!voice_ref_codes.empty()) {
                        result = this_tts->synthesize_with_embedding(
                            input, voice_embedding.data(), (int32_t)voice_embedding.size(),
                            params, voice_ref_codes.data(), voice_n_ref_frames, &sopts);
                    } else if (!voice_embedding.empty()) {
                        result = this_tts->synthesize_with_embedding(
                            input, voice_embedding.data(), (int32_t)voice_embedding.size(),
                            params, nullptr, 0, &sopts);
                    } else {
                        result = this_tts->synthesize(input, params, &sopts);
                    }

                    // ensure a header went out even if no pcm was produced, and
                    // flush any frames the muxer still holds.
                    ensure_header();
                    if (enc) {
                        emit(compressed_encoder_flush(enc));
                        compressed_encoder_close(enc);
                    }

                    if (is_sse) {
                        std::string done_frame = "event: speech.audio.done\ndata: "
                                               + build_done_event(result) + "\n\n";
                        sink.write(done_frame.data(), done_frame.size());
                    }
                    sink.done();
                    return false;
                });
            return;
        }

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

        // one-shot (no stream_format): preserve legacy behavior
        if (stream_format.empty()) {
            res.set_content(encode_format(response_format, result.audio, result.sample_rate),
                            content_type_for(response_format));
            return;
        }

        // stream_format=audio: raw chunked bytes in the chosen response_format.
        // wav uses a placeholder-size header so playback can start immediately;
        // pcm/mp3 need no container header so the body carries the full encoding.
        if (stream_format == "audio") {
            std::string header = (response_format == "wav")
                ? wav_streaming_header(result.sample_rate)
                : std::string();
            std::string body_bytes = (response_format == "wav")
                ? encode_pcm(result.audio)
                : encode_format(response_format, result.audio, result.sample_rate);
            const char * ctype = content_type_for(response_format);

            res.set_chunked_content_provider(ctype,
                [header = std::move(header), body_bytes = std::move(body_bytes)]
                (size_t /*offset*/, httplib::DataSink & sink) mutable -> bool {
                    if (!header.empty()) {
                        sink.write(header.data(), header.size());
                    }
                    sink.write(body_bytes.data(), body_bytes.size());
                    sink.done();
                    return false;
                });
            return;
        }

        // stream_format=sse: emit speech.audio.delta + speech.audio.done.
        // response_format still selects the bytes carried inside delta (wav, pcm,
        // or mp3). usage/timings on the done event are shaped to be consumed by
        // both openai clients and llama-swap's metrics_monitor.
        {
            std::string audio_bytes = encode_format(response_format, result.audio, result.sample_rate);

            json delta = {
                {"type", "speech.audio.delta"},
                {"audio", base64_encode(audio_bytes.data(), audio_bytes.size())},
            };
            std::string delta_frame = "event: speech.audio.delta\ndata: " + delta.dump() + "\n\n";
            std::string done_frame  = "event: speech.audio.done\ndata: "
                                    + build_done_event(result)
                                    + "\n\n";

            res.set_chunked_content_provider("text/event-stream",
                [delta_frame = std::move(delta_frame), done_frame = std::move(done_frame)]
                (size_t /*offset*/, httplib::DataSink & sink) mutable -> bool {
                    sink.write(delta_frame.data(), delta_frame.size());
                    sink.write(done_frame.data(),  done_frame.size());
                    sink.done();
                    return false;
                });
            return;
        }
    });

    fprintf(stderr, "server listening on %s:%d\n", sp.host.c_str(), sp.port);
    if (!svr.listen(sp.host, sp.port)) {
        fprintf(stderr, "fatal: failed to bind to %s:%d\n", sp.host.c_str(), sp.port);
        return 1;
    }

    return 0;
}
