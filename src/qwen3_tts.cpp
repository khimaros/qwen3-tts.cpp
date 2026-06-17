#include "qwen3_tts.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <atomic>

#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <sys/resource.h>
#endif

#ifdef QWEN3_TTS_LIBAV
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}
#endif

namespace qwen3_tts {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t phys_footprint_bytes = 0;
};

static bool get_process_memory_snapshot(process_memory_snapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic_info), &basic_count) != KERN_SUCCESS) {
        return false;
    }
    out.rss_bytes = (uint64_t) basic_info.resident_size;

    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.phys_footprint_bytes = (uint64_t) vm_info.phys_footprint;
    } else {
        out.phys_footprint_bytes = out.rss_bytes;
    }
    return true;
#else
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return false;
    }
    out.rss_bytes = (uint64_t) usage.ru_maxrss * 1024ULL;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#endif
}

static std::string format_bytes(uint64_t bytes) {
    static const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    double val = (double) bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

static void log_memory_usage(const char * label) {
    process_memory_snapshot mem;
    if (!get_process_memory_snapshot(mem)) {
        fprintf(stderr, "  [mem] %-24s unavailable\n", label);
        return;
    }
    fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
            label, format_bytes(mem.rss_bytes).c_str(),
            format_bytes(mem.phys_footprint_bytes).c_str());
}

static void resample_linear(const float * input, int input_len, int input_rate,
                            std::vector<float> & output, int output_rate) {
    double ratio = (double)input_rate / output_rate;
    int output_len = (int)((double)input_len / ratio);
    output.resize(output_len);
    
    for (int i = 0; i < output_len; ++i) {
        double src_idx = i * ratio;
        int idx0 = (int)src_idx;
        int idx1 = idx0 + 1;
        double frac = src_idx - idx0;
        
        if (idx1 >= input_len) {
            output[i] = input[input_len - 1];
        } else {
            output[i] = (float)((1.0 - frac) * input[idx0] + frac * input[idx1]);
        }
    }
}

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

bool Qwen3TTS::load_models(const std::string & model_dir) {
    // discover talker (any *.gguf not matching tokenizer) and vocoder (*tokenizer*.gguf).
    // prefer q8_0 over f16 for talker.
    namespace fs = std::filesystem;
    std::string tts_path, vocoder_path;
    std::string tts_q8, tts_f16, tts_other;
    std::error_code ec;
    for (const auto & entry : fs::directory_iterator(model_dir, ec)) {
        if (ec || !entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf") continue;
        std::string lower = name;
        for (auto & c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower.find("tokenizer") != std::string::npos) {
            vocoder_path = entry.path().string();
        } else if (lower.find("q8_0") != std::string::npos) {
            tts_q8 = entry.path().string();
        } else if (lower.find("f16") != std::string::npos || lower.find("fp16") != std::string::npos) {
            tts_f16 = entry.path().string();
        } else {
            tts_other = entry.path().string();
        }
    }
    tts_path = !tts_q8.empty() ? tts_q8 : (!tts_f16.empty() ? tts_f16 : tts_other);
    if (tts_path.empty() || vocoder_path.empty()) {
        error_msg_ = "could not find talker and tokenizer ggufs in " + model_dir;
        return false;
    }
    return load_model_files(tts_path, vocoder_path);
}

bool Qwen3TTS::load_model_files(const std::string & tts_path,
                                 const std::string & vocoder_path) {
    int64_t t_start = get_time_ms();
    log_memory_usage("load/start");

    transformer_.unload_model();
    audio_decoder_.unload_model();
    encoder_loaded_ = false;
    transformer_loaded_ = false;
    decoder_loaded_ = false;

    tts_model_path_ = tts_path;

    // derive vocoder path from same directory if not specified
    if (vocoder_path.empty()) {
        auto slash = tts_path.rfind('/');
        std::string dir = (slash != std::string::npos) ? tts_path.substr(0, slash) : ".";
        decoder_model_path_ = dir + "/qwen3-tts-tokenizer-f16.gguf";
    } else {
        decoder_model_path_ = vocoder_path;
    }

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_) {
        fprintf(stderr, "  Low-memory mode enabled (lazy decoder + component unloads)\n");
    }

    // Load TTS model (contains text tokenizer + transformer for generation)
    fprintf(stderr, "Loading TTS model from %s...\n", tts_model_path_.c_str());

    // Load text tokenizer from TTS model
    int64_t t_tokenizer_start = get_time_ms();
    {
        GGUFLoader loader;
        if (!loader.open(tts_model_path_)) {
            error_msg_ = "Failed to open TTS model: " + loader.get_error();
            return false;
        }

        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
            return false;
        }
        fprintf(stderr, "  Text tokenizer loaded: vocab_size=%d (%lld ms)\n",
                tokenizer_.get_config().vocab_size,
                (long long)(get_time_ms() - t_tokenizer_start));
    }
    log_memory_usage("load/after-tokenizer");

    // Speaker encoder is loaded lazily on first voice cloning request.
    fprintf(stderr, "  Speaker encoder: deferred (lazy load)\n");

    // Load TTS transformer from TTS model
    int64_t t_transformer_start = get_time_ms();
    if (!transformer_.load_model(tts_model_path_)) {
        error_msg_ = "Failed to load TTS transformer: " + transformer_.get_error();
        fprintf(stderr, "  ERROR: %s\n", error_msg_.c_str());
        return false;
    }
    transformer_loaded_ = true;
    fprintf(stderr, "  TTS transformer loaded: hidden_size=%d, n_layers=%d (%lld ms)\n",
            transformer_.get_config().hidden_size, transformer_.get_config().n_layers,
            (long long)(get_time_ms() - t_transformer_start));
    log_memory_usage("load/after-transformer");

    if (!low_mem_mode_) {
        // Load vocoder (audio decoder) from tokenizer model
        fprintf(stderr, "Loading vocoder from %s...\n", decoder_model_path_.c_str());
        int64_t t_decoder_start = get_time_ms();
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            error_msg_ = "Failed to load vocoder: " + audio_decoder_.get_error();
            fprintf(stderr, "  ERROR: %s\n", error_msg_.c_str());
            return false;
        }
        decoder_loaded_ = true;
        fprintf(stderr, "  Vocoder loaded: sample_rate=%d, n_codebooks=%d (%lld ms)\n",
                audio_decoder_.get_config().sample_rate, audio_decoder_.get_config().n_codebooks,
                (long long)(get_time_ms() - t_decoder_start));
        log_memory_usage("load/after-vocoder");
    } else {
        fprintf(stderr, "  Vocoder: deferred (lazy load)\n");
    }

    models_loaded_ = true;

    int64_t t_end = get_time_ms();
    fprintf(stderr, "All models loaded in %lld ms\n", (long long)(t_end - t_start));
    log_memory_usage("load/end");

    return true;
}

tts_result Qwen3TTS::synthesize(const std::string & text,
                                 const tts_params & params) {
    return synthesize(text, params, nullptr);
}

tts_result Qwen3TTS::synthesize(const std::string & text,
                                 const tts_params & params,
                                 const streaming_opts * stream) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    // For basic synthesis without voice cloning, we use a zero speaker embedding
    // This will use the model's default voice characteristics
    std::vector<float> zero_embedding(transformer_.get_config().hidden_size, 0.0f);

    return synthesize_internal(text, zero_embedding.data(), params, result,
                               nullptr, 0, stream);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const std::string & reference_audio,
                                            const tts_params & params) {
    tts_result result;
    
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        result.error_msg = "Failed to load reference audio: " + reference_audio;
        return result;
    }
    
    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }
    
    return synthesize_with_voice(text, ref_samples.data(), (int32_t)ref_samples.size(), params);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const float * ref_samples, int32_t n_ref_samples,
                                            const tts_params & params) {
    tts_result result;
    
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!encoder_loaded_) {
        if (tts_model_path_.empty()) {
            result.error_msg = "Internal error: missing TTS model path for lazy encoder load";
            return result;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return result;
        }
        audio_encoder_.set_abort_callback(abort_cb_, abort_data_);
        encoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_encoder_load_start));
            log_memory_usage("voice/after-encoder-load");
        }
    }

    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;

    if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
        result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return result;
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    if (params.print_progress) {
        fprintf(stderr, "Speaker embedding extracted: %zu floats\n", speaker_embedding.size());
    }

    // ICL mode: also encode reference audio to discrete speech codes
    if (!params.ref_text.empty()) {
        if (!codec_encoder_loaded_) {
            if (decoder_model_path_.empty()) {
                result.error_msg = "missing tokenizer model path for codec encoder";
                return result;
            }
            int64_t t0 = get_time_ms();
            if (!codec_encoder_.load_model(decoder_model_path_)) {
                result.error_msg = "failed to load codec encoder: " + codec_encoder_.get_error();
                return result;
            }
            codec_encoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(stderr, "  Codec encoder lazy-loaded in %lld ms\n",
                        (long long)(get_time_ms() - t0));
            }
        }

        int64_t t0 = get_time_ms();
        std::vector<int32_t> ref_codes;
        int32_t n_ref_frames = 0;
        if (!codec_encoder_.encode(ref_samples, n_ref_samples, ref_codes, n_ref_frames)) {
            result.error_msg = "failed to encode reference audio: " + codec_encoder_.get_error();
            return result;
        }
        if (params.print_progress) {
            fprintf(stderr, "Reference audio encoded: %d frames x 16 codebooks (ICL mode)\n", n_ref_frames);
        }
        if (params.print_timing) {
            fprintf(stderr, "  Codec encode: %lld ms\n", (long long)(get_time_ms() - t0));
        }

        return synthesize_internal(text, speaker_embedding.data(), params, result,
                                   ref_codes.data(), n_ref_frames);
    }

    return synthesize_internal(text, speaker_embedding.data(), params, result);
}

bool Qwen3TTS::extract_speaker_embedding(const std::string & reference_audio,
                                           std::vector<float> & embedding) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        error_msg_ = "Failed to load reference audio: " + reference_audio;
        return false;
    }

    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }

    if (!encoder_loaded_) {
        if (tts_model_path_.empty()) {
            error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
            return false;
        }
        if (!audio_encoder_.load_model(tts_model_path_)) {
            error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return false;
        }
        audio_encoder_.set_abort_callback(abort_cb_, abort_data_);
        encoder_loaded_ = true;
    }

    if (!audio_encoder_.encode(ref_samples.data(), (int32_t)ref_samples.size(), embedding)) {
        error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return false;
    }

    return true;
}

bool Qwen3TTS::encode_speech_codes(const float * samples, int32_t n_samples,
                                    std::vector<int32_t> & codes, int32_t & n_frames) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    if (!codec_encoder_loaded_) {
        if (decoder_model_path_.empty()) {
            error_msg_ = "missing tokenizer model path for codec encoder";
            return false;
        }
        if (!codec_encoder_.load_model(decoder_model_path_)) {
            error_msg_ = "failed to load codec encoder: " + codec_encoder_.get_error();
            return false;
        }
        codec_encoder_loaded_ = true;
    }

    if (!codec_encoder_.encode(samples, n_samples, codes, n_frames)) {
        error_msg_ = "failed to encode speech codes: " + codec_encoder_.get_error();
        return false;
    }

    return true;
}

tts_result Qwen3TTS::synthesize_with_embedding(const std::string & text,
                                                 const float * embedding, int32_t embedding_size,
                                                 const tts_params & params,
                                                 const int32_t * ref_codes,
                                                 int32_t n_ref_frames,
                                                 const streaming_opts * stream) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!embedding) {
        result.error_msg = "Speaker embedding is null";
        return result;
    }

    const int32_t expected_size = transformer_.get_config().hidden_size;
    if (embedding_size != expected_size) {
        result.error_msg = "Invalid embedding size: expected " + std::to_string(expected_size)
                         + ", got " + std::to_string(embedding_size);
        return result;
    }

    return synthesize_internal(text, embedding, params, result, ref_codes, n_ref_frames, stream);
}

tts_result Qwen3TTS::synthesize_internal(const std::string & text,
                                          const float * speaker_embedding,
                                          const tts_params & params,
                                          tts_result & result,
                                          const int32_t * ref_codes,
                                          int32_t n_ref_frames,
                                          const streaming_opts * stream) {
    int64_t t_total_start = get_time_ms();
    auto sample_memory = [&](const char * stage) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) {
            return;
        }
        if (result.mem_rss_start_bytes == 0) {
            result.mem_rss_start_bytes = mem.rss_bytes;
            result.mem_phys_start_bytes = mem.phys_footprint_bytes;
        }
        result.mem_rss_end_bytes = mem.rss_bytes;
        result.mem_phys_end_bytes = mem.phys_footprint_bytes;
        if (mem.rss_bytes > result.mem_rss_peak_bytes) {
            result.mem_rss_peak_bytes = mem.rss_bytes;
        }
        if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes) {
            result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
        }
        if (params.print_timing) {
            fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
                    stage,
                    format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str());
        }
    };
    sample_memory("synth/start");
    
    // Step 2: Tokenize input text and optional instruction prompt
    int64_t t_tokenize_start = get_time_ms();
    std::vector<int32_t> text_tokens = tokenizer_.encode_for_tts(text);
    std::vector<int32_t> instruct_tokens;
    if (!params.instructions.empty() && transformer_.get_config().model_size != "0b6") {
        instruct_tokens = tokenizer_.encode_instruct(params.instructions);
    }
    result.t_tokenize_ms = get_time_ms() - t_tokenize_start;
    result.n_text_tokens = (int32_t)text_tokens.size();
    sample_memory("synth/after-tokenize");

    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    
    if (params.print_progress) {
        fprintf(stderr, "Text tokenized: %zu tokens\n", text_tokens.size());
        fprintf(stderr, "  Tokens: ");
        for (size_t i = 0; i < std::min(text_tokens.size(), (size_t)10); ++i) {
            fprintf(stderr, "%d ", text_tokens[i]);
        }
        if (text_tokens.size() > 10) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }
    
    // Step 3: Generate speech codes using TTS transformer
    int64_t t_generate_start = get_time_ms();
    if (!transformer_loaded_) {
        int64_t t_reload_start = get_time_ms();
        if (!transformer_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to reload TTS transformer: " + transformer_.get_error();
            return result;
        }
        transformer_.set_abort_callback(abort_cb_, abort_data_);
        transformer_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Transformer reloaded in %lld ms\n",
                    (long long)(get_time_ms() - t_reload_start));
            sample_memory("synth/after-transformer-reload");
        }
    }
    transformer_.clear_kv_cache();
    transformer_.set_verbose(params.print_progress);
    if (params.seed >= 0) {
        transformer_.set_seed((uint64_t)params.seed);
    }

    // tokenize ref_text for ICL mode
    std::vector<int32_t> ref_text_tokens;
    if (ref_codes && n_ref_frames > 0 && !params.ref_text.empty()) {
        ref_text_tokens = tokenizer_.encode_for_tts(params.ref_text);
        // python _build_ref_text wraps as <|im_start|>assistant\n{text}<|im_end|>\n
        // and slices ref_id[:, 3:-2], yielding just the content tokens. our
        // encode_for_tts uses the longer assistant wrap with a trailing
        // <|im_start|>assistant\n (8 framing tokens), so we drop 3 prefix + 5
        // suffix tokens to land on the same content-only window. leaving the
        // boundary tokens in causes the talker to treat ref+new as separate
        // turns, producing hangs and ref-text interjection.
        if ((int)ref_text_tokens.size() > 8) {
            ref_text_tokens = std::vector<int32_t>(
                ref_text_tokens.begin() + 3,
                ref_text_tokens.end() - 5
            );
        } else {
            ref_text_tokens.clear();
        }
    }

    // streaming: install per-frame callback that batches codes and live-decodes
    // via the vocoder's streaming path. we must also ensure the decoder is
    // loaded before generate() so its stream_decode can be called from within
    // the callback. ICL warm-up (ref_codes) is fed as a discarded chunk below.
    const bool streaming = stream && stream->batch_size > 0 && stream->on_pcm;
    std::vector<int32_t> stream_buf;
    size_t stream_cb_count = 0;
    bool stream_cb_aborted = false;
    if (streaming) {
        if (!decoder_loaded_) {
            int64_t t_decoder_load_start = get_time_ms();
            if (decoder_model_path_.empty()) {
                result.error_msg = "Internal error: missing vocoder model path";
                return result;
            }
            if (!audio_decoder_.load_model(decoder_model_path_)) {
                result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
                return result;
            }
            audio_decoder_.set_abort_callback(abort_cb_, abort_data_);
            decoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(stderr, "  Vocoder lazy-loaded in %lld ms\n",
                        (long long)(get_time_ms() - t_decoder_load_start));
                sample_memory("synth/after-vocoder-load-stream");
            }
        }
        audio_decoder_.stream_reset();
        const int n_cb = transformer_.get_config().n_codebooks;

        // ICL warm-up: feed ref_codes through the streaming decoder and
        // discard its PCM. mirrors the non-streaming prepend+trim path.
        if (ref_codes && n_ref_frames > 0 && !params.ref_text.empty()) {
            std::vector<float> warmup_pcm;
            if (!audio_decoder_.stream_decode(ref_codes, n_ref_frames, warmup_pcm)) {
                result.error_msg = "Failed to warm-up vocoder with ref codes: " + audio_decoder_.get_error();
                return result;
            }
            // discard warmup_pcm — downstream only sees post-ref PCM
        }

        stream_buf.reserve((size_t) stream->batch_size * n_cb);
        transformer_.set_frame_callback(
            [this, stream, &stream_buf, &stream_cb_count, &stream_cb_aborted, n_cb, &result]
            (int32_t /*frame_idx*/, const int32_t * frame_codes) -> bool {
                for (int c = 0; c < n_cb; ++c) stream_buf.push_back(frame_codes[c]);
                const int frames_buffered = (int) (stream_buf.size() / n_cb);
                if (frames_buffered >= stream->batch_size) {
                    std::vector<float> pcm;
                    if (!audio_decoder_.stream_decode(stream_buf.data(), frames_buffered, pcm)) {
                        stream_cb_aborted = true;
                        return false;
                    }
                    stream_buf.clear();
                    result.audio.insert(result.audio.end(), pcm.begin(), pcm.end());
                    stream_cb_count++;
                    if (!stream->on_pcm(pcm.data(), pcm.size())) {
                        stream_cb_aborted = true;
                        return false;
                    }
                }
                return true;
            });
    }

    std::vector<int32_t> speech_codes;
    if (!transformer_.generate(text_tokens.data(), (int32_t)text_tokens.size(),
                               speaker_embedding, params.max_audio_tokens, speech_codes,
                               params.language_id, params.repetition_penalty,
                               params.temperature, params.top_k,
                               instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                               (int32_t)instruct_tokens.size(),
                               ref_text_tokens.empty() ? nullptr : ref_text_tokens.data(),
                               (int32_t)ref_text_tokens.size(),
                               ref_codes, n_ref_frames)) {
        if (streaming) transformer_.set_frame_callback({});
        result.error_msg = "Failed to generate speech codes: " + transformer_.get_error();
        return result;
    }
    if (streaming) {
        transformer_.set_frame_callback({});
    }
    result.t_generate_ms = get_time_ms() - t_generate_start;
    result.n_prefill_tokens = transformer_.get_last_n_prefill_tokens();
    result.t_prefill_ms     = transformer_.get_last_prefill_ms();
    sample_memory("synth/after-generate");

    if (is_aborted()) {
        result.error_msg = "Aborted";
        return result;
    }

    int n_codebooks = transformer_.get_config().n_codebooks;
    int n_frames = (int)speech_codes.size() / n_codebooks;
    result.n_audio_tokens = n_frames;

    if (params.print_progress) {
        fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
    }
    
    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }

    if (low_mem_mode_) {
        transformer_.unload_model();
        transformer_loaded_ = false;
        sample_memory("synth/after-transformer-unload");
    }
    
    // Step 4: Decode speech codes to waveform using vocoder
    int64_t t_decode_start = get_time_ms();

    // streaming path: frames were already decoded live in the frame callback.
    // flush any residual (< batch_size) frames now, then skip the one-shot
    // decode that follows.
    if (streaming) {
        const int n_cb = transformer_.get_config().n_codebooks;
        const int leftover = (int) (stream_buf.size() / n_cb);
        if (leftover > 0 && !stream_cb_aborted) {
            std::vector<float> pcm;
            if (!audio_decoder_.stream_decode(stream_buf.data(), leftover, pcm)) {
                result.error_msg = "Failed to flush streaming vocoder: " + audio_decoder_.get_error();
                return result;
            }
            stream_buf.clear();
            result.audio.insert(result.audio.end(), pcm.begin(), pcm.end());
            if (!stream->on_pcm(pcm.data(), pcm.size())) {
                stream_cb_aborted = true;
            }
        }
        result.t_decode_ms = get_time_ms() - t_decode_start;
        sample_memory("synth/after-stream-decode");
        if (params.print_progress) {
            fprintf(stderr, "Streaming: %zu batches dispatched, %zu total samples\n",
                    stream_cb_count, result.audio.size());
        }
        result.sample_rate = audio_decoder_.get_config().sample_rate;
        result.success = !stream_cb_aborted;
        if (stream_cb_aborted && result.error_msg.empty()) {
            result.error_msg = "Streaming consumer aborted";
        }
        result.t_total_ms = get_time_ms() - t_total_start;
        sample_memory("synth/end");
        return result;
    }

    if (!decoder_loaded_) {
        int64_t t_decoder_load_start = get_time_ms();
        if (decoder_model_path_.empty()) {
            result.error_msg = "Internal error: missing vocoder model path";
            return result;
        }
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
            return result;
        }
        audio_decoder_.set_abort_callback(abort_cb_, abort_data_);
        decoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Vocoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_decoder_load_start));
            sample_memory("synth/after-vocoder-load");
        }
    }
    
    // ICL: prepend ref_codes to the talker output so the vocoder has warm
    // context (matches qwen3_tts_model.py:616, torch.cat([ref, new])). We then
    // slice the ref portion off the decoded wav below. Without this, the
    // vocoder cold-starts and produces ~350ms of noise at the beginning.
    std::vector<int32_t> codes_for_decode;
    int32_t total_frames = n_frames;
    if (ref_codes && n_ref_frames > 0 && !params.ref_text.empty()) {
        total_frames = n_ref_frames + n_frames;
        codes_for_decode.resize((size_t)total_frames * n_codebooks);
        std::memcpy(codes_for_decode.data(),
                    ref_codes,
                    (size_t)n_ref_frames * n_codebooks * sizeof(int32_t));
        std::memcpy(codes_for_decode.data() + (size_t)n_ref_frames * n_codebooks,
                    speech_codes.data(),
                    speech_codes.size() * sizeof(int32_t));
    }
    const int32_t * decode_codes =
        codes_for_decode.empty() ? speech_codes.data() : codes_for_decode.data();
    fprintf(stderr, "  [icl] ref_frames=%d new_frames=%d total_frames=%d prepended=%d\n",
            n_ref_frames, n_frames, total_frames, (int)!codes_for_decode.empty());
    if (const char * dp = std::getenv("QWEN3_TTS_DUMP_CODES")) {
        static std::atomic<int> dump_counter{0};
        int idx = dump_counter.fetch_add(1);
        std::string path = std::string(dp);
        if (path.find("%d") != std::string::npos) {
            char buf[1024];
            snprintf(buf, sizeof(buf), dp, idx);
            path = buf;
        }
        FILE * fp = fopen(path.c_str(), "wb");
        if (fp) {
            int32_t hdr[3] = { n_ref_frames, n_frames, n_codebooks };
            fwrite(hdr, sizeof(int32_t), 3, fp);
            if (ref_codes && n_ref_frames > 0) {
                fwrite(ref_codes, sizeof(int32_t), (size_t)n_ref_frames * n_codebooks, fp);
            }
            fwrite(speech_codes.data(), sizeof(int32_t), speech_codes.size(), fp);
            fclose(fp);
            fprintf(stderr, "  dumped ref+new codes to %s\n", path.c_str());
        }
    }

    if (!audio_decoder_.decode(decode_codes, total_frames, result.audio)) {
        result.error_msg = "Failed to decode speech codes: " + audio_decoder_.get_error();
        return result;
    }

    // trim the ref-code portion from the decoded wav (qwen3_tts_model.py:628).
    if (!codes_for_decode.empty() && !result.audio.empty()) {
        size_t total_samples = result.audio.size();
        size_t cut = (size_t)(((int64_t)n_ref_frames * (int64_t)total_samples)
                               / (int64_t)total_frames);
        if (cut < total_samples) {
            result.audio.erase(result.audio.begin(),
                               result.audio.begin() + (ptrdiff_t)cut);
        }
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    sample_memory("synth/after-decode");

    if (low_mem_mode_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
        sample_memory("synth/after-vocoder-unload");
    }
    
    result.sample_rate = audio_decoder_.get_config().sample_rate;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;
    sample_memory("synth/end");
    
    if (params.print_timing) {
        const double audio_sec = result.sample_rate > 0
            ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
        const double wall_sec = (double) result.t_total_ms / 1000.0;
        const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
        const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenization:    %lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Speaker encode:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Code generation: %lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Vocoder decode:  %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Audio duration:  %.2f s\n", audio_sec);
        fprintf(stderr, "  Throughput:      %.2fx realtime (RTF=%.3f)\n", x_realtime, realtime_factor);
        fprintf(stderr, "\nMemory:\n");
        fprintf(stderr, "  RSS start/end:   %s -> %s\n",
                format_bytes(result.mem_rss_start_bytes).c_str(),
                format_bytes(result.mem_rss_end_bytes).c_str());
        fprintf(stderr, "  RSS peak:        %s\n",
                format_bytes(result.mem_rss_peak_bytes).c_str());
        fprintf(stderr, "  Phys start/end:  %s -> %s\n",
                format_bytes(result.mem_phys_start_bytes).c_str(),
                format_bytes(result.mem_phys_end_bytes).c_str());
        fprintf(stderr, "  Phys peak:       %s\n",
                format_bytes(result.mem_phys_peak_bytes).c_str());
    }
    
    return result;
}

int32_t Qwen3TTS::get_hidden_size() const {
    return transformer_.get_config().hidden_size;
}

const std::string & Qwen3TTS::get_model_type() const {
    return transformer_.get_config().model_type;
}

const std::vector<std::string> & Qwen3TTS::get_speaker_names() const {
    return transformer_.get_config().speaker_names;
}

const std::vector<int32_t> & Qwen3TTS::get_speaker_ids() const {
    return transformer_.get_config().speaker_ids;
}

bool Qwen3TTS::has_speaker_encoder() const {
    return transformer_.get_config().has_speaker_encoder;
}

int32_t Qwen3TTS::get_speaker_id(const std::string & name) const {
    auto & names = transformer_.get_config().speaker_names;
    auto & ids = transformer_.get_config().speaker_ids;
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] == name) return ids[i];
    }
    return -1;
}

bool Qwen3TTS::get_speaker_embedding(const std::string & name, std::vector<float> & embedding) {
    int32_t spk_id = get_speaker_id(name);
    if (spk_id < 0) {
        error_msg_ = "unknown speaker: " + name;
        return false;
    }
    if (!transformer_.get_codec_embedding(spk_id, embedding)) {
        error_msg_ = "failed to get speaker embedding: " + transformer_.get_error();
        return false;
    }
    return true;
}

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

void Qwen3TTS::set_abort_callback(ggml_abort_callback callback, void * data) {
    abort_cb_ = callback;
    abort_data_ = data;
    transformer_.set_abort_callback(callback, data);
    audio_encoder_.set_abort_callback(callback, data);
    audio_decoder_.set_abort_callback(callback, data);
}

// Mix an interleaved multi-channel buffer down to mono, storing results in `out`.
// InputT must be convertible to float; `scale` is applied before summing.
template<typename InputT>
static void mix_to_mono(const InputT * interleaved, int n_samples, int num_channels,
                        float scale, std::vector<float> & out) {
    out.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < num_channels; ++c) {
            sum += static_cast<float>(interleaved[i * num_channels + c]) * scale;
        }
        out[i] = sum / num_channels;
    }
}

// Get lowercase file extension from path
static std::string get_file_extension(const std::string & path) {
    auto dot_pos = path.rfind('.');
    if (dot_pos == std::string::npos) return "";
    std::string ext = path.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

// WAV file loading (16-bit PCM, 32-bit PCM, or 32-bit IEEE float)
static bool load_wav_file(const std::string & path, std::vector<float> & samples,
                          int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }

    // Read RIFF header
    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "ERROR: Not a RIFF file\n");
        fclose(f);
        return false;
    }

    uint32_t file_size;
    if (fread(&file_size, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }

    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Not a WAVE file\n");
        fclose(f);
        return false;
    }

    // Find fmt and data chunks
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;

    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;

        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, f) != 1) break;
            if (fread(&num_channels, 2, 1, f) != 1) break;
            if (fread(&sr, 4, 1, f) != 1) break;
            fseek(f, 6, SEEK_CUR);  // Skip byte rate and block align
            if (fread(&bits_per_sample, 2, 1, f) != 1) break;

            // Handle WAVE_FORMAT_EXTENSIBLE: read actual format from SubFormat GUID
            if (audio_format == 0xFFFE && chunk_size >= 40) {
                fseek(f, 8, SEEK_CUR);  // Skip cbSize(2) + validBitsPerSample(2) + channelMask(4)
                uint16_t sub_format = 0;
                if (fread(&sub_format, 2, 1, f) != 1) break;
                audio_format = sub_format;
                // Skip remaining SubFormat GUID bytes and any extra data
                fseek(f, chunk_size - 26, SEEK_CUR);
            }
            // Skip any extra format bytes for non-extensible formats
            else if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        }
        else if (strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = sr;

            if (audio_format == 1) {  // PCM
                if (bits_per_sample == 16) {
                    int n_samples = chunk_size / (2 * num_channels);
                    std::vector<int16_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 2, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    mix_to_mono(raw.data(), n_samples, num_channels, 1.0f / 32768.0f, samples);
                }
                else if (bits_per_sample == 32) {
                    int n_samples = chunk_size / (4 * num_channels);
                    std::vector<int32_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    mix_to_mono(raw.data(), n_samples, num_channels, 1.0f / 2147483648.0f, samples);
                }
                else {
                    fprintf(stderr, "ERROR: Unsupported bits per sample: %d\n", bits_per_sample);
                    fclose(f);
                    return false;
                }
            }
            else if (audio_format == 3) {  // IEEE float
                int n_samples = chunk_size / (4 * num_channels);
                std::vector<float> raw(n_samples * num_channels);
                if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                    fclose(f);
                    return false;
                }
                mix_to_mono(raw.data(), n_samples, num_channels, 1.0f, samples);
            }
            else {
                fprintf(stderr, "ERROR: Unsupported audio format: %d\n", audio_format);
                fclose(f);
                return false;
            }

            fclose(f);
            return true;
        }
        else {
            // Skip unknown chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    fprintf(stderr, "ERROR: No data chunk found\n");
    fclose(f);
    return false;
}

// Audio file loading - dispatches to format-specific loaders based on file extension
bool load_audio_file(const std::string & path, std::vector<float> & samples,
                     int & sample_rate) {
    std::string ext = get_file_extension(path);

    if (ext == ".wav") {
        return load_wav_file(path, samples, sample_rate);
    } else {
        fprintf(stderr, "ERROR: Unsupported audio format '%s'. Supported formats: .wav\n", ext.c_str());
        return false;
    }
}

// WAV file saving (16-bit PCM at specified sample rate)
static bool save_wav_file(const std::string & path, const std::vector<float> & samples,
                          int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create WAV file: %s\n", path.c_str());
        return false;
    }
    
    // WAV header parameters
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;
    uint32_t data_size = samples.size() * block_align;
    uint32_t file_size = 36 + data_size;
    
    // Write RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    
    // Write fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = sample_rate;
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    
    // Write data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    
    // Convert float samples to 16-bit PCM and write
    for (size_t i = 0; i < samples.size(); ++i) {
        // Clamp to [-1, 1] and convert to int16
        float sample = samples[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t pcm_sample = (int16_t)(sample * 32767.0f);
        fwrite(&pcm_sample, 2, 1, f);
    }

    fclose(f);
    return true;
}

bool compressed_audio_supported() {
#ifdef QWEN3_TTS_LIBAV
    return true;
#else
    return false;
#endif
}

bool codec_from_name(const std::string & name, audio_codec & out) {
    if (name == "mp3") { out = audio_codec::mp3; return true; }
    if (name == "opus" || name == "ogg") { out = audio_codec::opus; return true; }
    return false;
}

#ifdef QWEN3_TTS_LIBAV

// in-memory libav muxing: the encoder feeds float pcm through a resampler into a
// frame-size fifo, encodes full frames, and muxes packets to a custom non-
// seekable AVIO whose write callback appends to `pending`. callers drain
// `pending` from the _write/_flush return values, so the same path serves both
// one-shot and streaming output. mp3 disables the xing/id3 tags so no seek is
// needed; ogg/opus is streamable by nature.
static constexpr int LIBAV_IO_BUFSZ = 4096;

struct compressed_encoder {
    AVFormatContext * fmt   = nullptr;
    AVCodecContext  * cctx  = nullptr;
    AVStream        * st    = nullptr;
    AVIOContext     * avio  = nullptr;
    SwrContext      * swr   = nullptr;
    AVAudioFifo     * fifo  = nullptr;
    AVFrame         * frame = nullptr;
    AVPacket        * pkt   = nullptr;
    std::string       pending;
    int               frame_size = 0;
    int64_t           pts = 0;
};

static int libav_write_cb(void * opaque, const uint8_t * buf, int size) {
    auto * e = static_cast<compressed_encoder *>(opaque);
    e->pending.append(reinterpret_cast<const char *>(buf), (size_t)size);
    return size;
}

static std::string take_pending(compressed_encoder * e) {
    std::string out;
    out.swap(e->pending);
    return out;
}

// pull encoded packets and mux them; returns 0 or a negative libav error.
static int libav_mux_packets(compressed_encoder * e) {
    for (;;) {
        int r = avcodec_receive_packet(e->cctx, e->pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return 0;
        if (r < 0) return r;
        e->pkt->stream_index = e->st->index;
        av_packet_rescale_ts(e->pkt, e->cctx->time_base, e->st->time_base);
        r = av_interleaved_write_frame(e->fmt, e->pkt);
        av_packet_unref(e->pkt);
        if (r < 0) return r;
    }
}

// encode whole frames from the fifo; on final, pad the tail frame and drain.
static int libav_encode_fifo(compressed_encoder * e, bool final_flush) {
    const int fs = e->frame_size;
    const int ch = e->cctx->ch_layout.nb_channels;
    while (av_audio_fifo_size(e->fifo) >= fs) {
        if (av_frame_make_writable(e->frame) < 0) return -1;
        av_audio_fifo_read(e->fifo, (void **)e->frame->data, fs);
        e->frame->nb_samples = fs;
        e->frame->pts = e->pts; e->pts += fs;
        int r = avcodec_send_frame(e->cctx, e->frame);
        if (r < 0 || (r = libav_mux_packets(e)) < 0) return r;
    }
    if (!final_flush) return 0;

    int rem = av_audio_fifo_size(e->fifo);
    if (rem > 0) {
        if (av_frame_make_writable(e->frame) < 0) return -1;
        av_audio_fifo_read(e->fifo, (void **)e->frame->data, rem);
        if (rem < fs)  // pad to a full frame: fixed-size encoders reject partials
            av_samples_set_silence(e->frame->data, rem, fs - rem, ch, e->cctx->sample_fmt);
        e->frame->nb_samples = fs;
        e->frame->pts = e->pts; e->pts += rem;
        int r = avcodec_send_frame(e->cctx, e->frame);
        if (r < 0 || (r = libav_mux_packets(e)) < 0) return r;
    }
    avcodec_send_frame(e->cctx, nullptr);  // flush the encoder
    return libav_mux_packets(e);
}

compressed_encoder * compressed_encoder_open(audio_codec codec, int sample_rate) {
    av_log_set_level(AV_LOG_ERROR);

    const char * muxer; const char * enc_name; enum AVCodecID fallback; int bitrate;
    if (codec == audio_codec::mp3) {
        muxer = "mp3";  enc_name = "libmp3lame"; fallback = AV_CODEC_ID_MP3;  bitrate = MP3_BITRATE;
    } else {
        muxer = "ogg";  enc_name = "libopus";    fallback = AV_CODEC_ID_OPUS; bitrate = OPUS_BITRATE;
    }

    const AVCodec * encoder = avcodec_find_encoder_by_name(enc_name);
    if (!encoder) encoder = avcodec_find_encoder(fallback);
    if (!encoder) return nullptr;

    auto * e = new compressed_encoder();
    if (avformat_alloc_output_context2(&e->fmt, nullptr, muxer, nullptr) < 0 || !e->fmt) {
        compressed_encoder_close(e); return nullptr;
    }

    unsigned char * iobuf = (unsigned char *)av_malloc(LIBAV_IO_BUFSZ);
    if (!iobuf) { compressed_encoder_close(e); return nullptr; }
    e->avio = avio_alloc_context(iobuf, LIBAV_IO_BUFSZ, 1, e, nullptr, libav_write_cb, nullptr);
    if (!e->avio) { av_free(iobuf); compressed_encoder_close(e); return nullptr; }
    e->avio->seekable = 0;
    e->fmt->pb = e->avio;
    e->fmt->flags |= AVFMT_FLAG_CUSTOM_IO;

    e->cctx = avcodec_alloc_context3(encoder);
    if (!e->cctx) { compressed_encoder_close(e); return nullptr; }

    // pick a sample format the encoder supports, preferring float.
    const enum AVSampleFormat * sfmts = nullptr;
    int n_sfmts = 0;
    avcodec_get_supported_config(nullptr, encoder, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0,
                                 (const void **)&sfmts, &n_sfmts);
    enum AVSampleFormat want = (sfmts && n_sfmts > 0) ? sfmts[0] : AV_SAMPLE_FMT_FLTP;
    for (int i = 0; sfmts && i < n_sfmts; i++) {
        if (sfmts[i] == AV_SAMPLE_FMT_FLT || sfmts[i] == AV_SAMPLE_FMT_FLTP) { want = sfmts[i]; break; }
    }

    e->cctx->sample_fmt  = want;
    e->cctx->sample_rate = sample_rate;
    av_channel_layout_default(&e->cctx->ch_layout, 1);  // mono
    e->cctx->bit_rate  = bitrate;
    e->cctx->time_base = av_make_q(1, sample_rate);
    if (e->fmt->oformat->flags & AVFMT_GLOBALHEADER)
        e->cctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(e->cctx, encoder, nullptr) < 0) { compressed_encoder_close(e); return nullptr; }
    e->frame_size = e->cctx->frame_size > 0 ? e->cctx->frame_size : 1024;

    e->st = avformat_new_stream(e->fmt, nullptr);
    if (!e->st || avcodec_parameters_from_context(e->st->codecpar, e->cctx) < 0) {
        compressed_encoder_close(e); return nullptr;
    }
    e->st->time_base = e->cctx->time_base;

    AVChannelLayout in_layout;
    av_channel_layout_default(&in_layout, 1);
    int r = swr_alloc_set_opts2(&e->swr,
        &e->cctx->ch_layout, e->cctx->sample_fmt, e->cctx->sample_rate,
        &in_layout, AV_SAMPLE_FMT_FLT, sample_rate, 0, nullptr);
    av_channel_layout_uninit(&in_layout);
    if (r < 0 || swr_init(e->swr) < 0) { compressed_encoder_close(e); return nullptr; }

    e->fifo = av_audio_fifo_alloc(e->cctx->sample_fmt, e->cctx->ch_layout.nb_channels, 1);
    e->frame = av_frame_alloc();
    e->pkt = av_packet_alloc();
    if (!e->fifo || !e->frame || !e->pkt) { compressed_encoder_close(e); return nullptr; }
    e->frame->format = e->cctx->sample_fmt;
    e->frame->sample_rate = e->cctx->sample_rate;
    e->frame->nb_samples = e->frame_size;
    av_channel_layout_copy(&e->frame->ch_layout, &e->cctx->ch_layout);
    if (av_frame_get_buffer(e->frame, 0) < 0) { compressed_encoder_close(e); return nullptr; }

    AVDictionary * mux_opts = nullptr;
    if (codec == audio_codec::mp3) {
        av_dict_set(&mux_opts, "write_xing", "0", 0);     // no xing tag => no seek
        av_dict_set(&mux_opts, "id3v2_version", "0", 0);  // raw frames
    }
    r = avformat_write_header(e->fmt, &mux_opts);
    av_dict_free(&mux_opts);
    if (r < 0) { compressed_encoder_close(e); return nullptr; }
    return e;
}

std::string compressed_encoder_write(compressed_encoder * e, const float * pcm, size_t n) {
    if (!e) return {};
    if (pcm && n > 0) {
        const uint8_t * in_data[1] = { reinterpret_cast<const uint8_t *>(pcm) };
        int out_max = (int)swr_get_out_samples(e->swr, (int)n);
        if (out_max < (int)n) out_max = (int)n;
        uint8_t ** conv = nullptr;
        if (av_samples_alloc_array_and_samples(&conv, nullptr, e->cctx->ch_layout.nb_channels,
                                               out_max, e->cctx->sample_fmt, 0) >= 0) {
            int got = swr_convert(e->swr, conv, out_max, in_data, (int)n);
            if (got > 0) av_audio_fifo_write(e->fifo, (void **)conv, got);
            av_freep(&conv[0]);
            av_freep(&conv);
        }
        libav_encode_fifo(e, false);
    }
    return take_pending(e);
}

std::string compressed_encoder_flush(compressed_encoder * e) {
    if (!e) return {};
    // drain any samples buffered inside the resampler, then encode + finalize.
    int out_max = (int)swr_get_out_samples(e->swr, 0);
    if (out_max > 0) {
        uint8_t ** conv = nullptr;
        if (av_samples_alloc_array_and_samples(&conv, nullptr, e->cctx->ch_layout.nb_channels,
                                               out_max, e->cctx->sample_fmt, 0) >= 0) {
            int got = swr_convert(e->swr, conv, out_max, nullptr, 0);
            if (got > 0) av_audio_fifo_write(e->fifo, (void **)conv, got);
            av_freep(&conv[0]);
            av_freep(&conv);
        }
    }
    libav_encode_fifo(e, true);
    av_write_trailer(e->fmt);
    return take_pending(e);
}

void compressed_encoder_close(compressed_encoder * e) {
    if (!e) return;
    if (e->frame) av_frame_free(&e->frame);
    if (e->pkt)   av_packet_free(&e->pkt);
    if (e->fifo)  av_audio_fifo_free(e->fifo);
    if (e->swr)   swr_free(&e->swr);
    if (e->cctx)  avcodec_free_context(&e->cctx);
    if (e->fmt)   avformat_free_context(e->fmt);  // does not free custom pb
    if (e->avio)  { av_freep(&e->avio->buffer); avio_context_free(&e->avio); }
    delete e;
}

#else  // libav not compiled in: stubs so callers link regardless

compressed_encoder * compressed_encoder_open(audio_codec, int) { return nullptr; }
std::string compressed_encoder_write(compressed_encoder *, const float *, size_t) { return {}; }
std::string compressed_encoder_flush(compressed_encoder *) { return {}; }
void        compressed_encoder_close(compressed_encoder *) {}

#endif // QWEN3_TTS_LIBAV

std::string encode_compressed(audio_codec codec, const std::vector<float> & samples,
                              int sample_rate) {
    compressed_encoder * e = compressed_encoder_open(codec, sample_rate);
    if (!e) return {};
    std::string out = compressed_encoder_write(e, samples.data(), samples.size());
    out += compressed_encoder_flush(e);
    compressed_encoder_close(e);
    return out;
}

// save_audio_file dispatches on the path extension: ".mp3"/".opus"/".ogg" emit
// compressed audio, anything else emits 16-bit WAV.
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate) {
    std::string ext = get_file_extension(path);
    audio_codec codec;
    if (ext.size() > 1 && codec_from_name(ext.substr(1), codec)) {
        const char * name = ext.c_str() + 1;
        if (!compressed_audio_supported()) {
            fprintf(stderr, "ERROR: %s output not supported by this build "
                            "(rebuild with libav/ffmpeg dev libraries)\n", name);
            return false;
        }
        std::string data = encode_compressed(codec, samples, sample_rate);
        if (data.empty()) {
            fprintf(stderr, "ERROR: %s encoding failed\n", name);
            return false;
        }
        FILE * f = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "ERROR: Cannot create file: %s\n", path.c_str());
            return false;
        }
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
        return true;
    }
    return save_wav_file(path, samples, sample_rate);
}

} // namespace qwen3_tts
