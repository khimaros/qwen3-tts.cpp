#pragma once

#include "text_tokenizer.h"
#include "tts_transformer.h"
#include "audio_tokenizer_encoder.h"
#include "audio_codec_encoder.h"
#include "audio_tokenizer_decoder.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <cstddef>

namespace qwen3_tts {

// TTS generation parameters
struct tts_params {
    // Maximum number of audio tokens to generate
    int32_t max_audio_tokens = 2048;
    
    // Temperature for sampling (0 = greedy)
    float temperature = 0.9f;
    
    // Top-p sampling
    float top_p = 1.0f;
    
    // Top-k sampling (0 = disabled)
    int32_t top_k = 50;
    
    // Number of threads
    int32_t n_threads = 4;
    
    // Print progress during generation
    bool print_progress = false;
    
    // Print timing information
    bool print_timing = true;
    
    // Repetition penalty for CB0 token generation (HuggingFace style)
    float repetition_penalty = 1.05f;

    // Language ID for codec (2050=en, 2069=ru, 2055=zh, 2058=ja, 2064=ko, 2053=de, 2061=fr, 2054=es)
    int32_t language_id = 2050;

    // Voice steering instruction (e.g. "speak slowly in a calm tone")
    std::string instructions;

    // Reference text for ICL voice cloning (when set, enables ICL mode instead of x-vector)
    std::string ref_text;

    // Sampling seed. < 0 means leave RNG as-is (non-deterministic); >= 0 reseeds
    // the transformer's RNG so runs are reproducible.
    int64_t seed = -1;
};

// TTS generation result
struct tts_result {
    // Generated audio samples (24kHz, mono)
    std::vector<float> audio;
    
    // Sample rate
    int32_t sample_rate = 24000;
    
    // Success flag
    bool success = false;
    
    // Error message if failed
    std::string error_msg;
    
    // Token counts (real, not approximated)
    int32_t n_text_tokens = 0;      // user-content text tokens (maps to openai usage.input_tokens)
    int32_t n_prefill_tokens = 0;   // total positions the transformer prefilled
                                    // (text + instruct + ref_text + ref_codes + framing)
    int32_t n_audio_tokens = 0;     // codec frames produced by the transformer

    // Timing info (in milliseconds)
    int64_t t_load_ms = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms = 0;        // speaker encoder (voice cloning)
    int64_t t_generate_ms = 0;      // full transformer generate() wall time
    int64_t t_prefill_ms = 0;       // subset of t_generate_ms: build_prefill + forward_prefill
    int64_t t_decode_ms = 0;        // vocoder decode
    int64_t t_total_ms = 0;

    // Process memory snapshots (bytes)
    uint64_t mem_rss_start_bytes = 0;
    uint64_t mem_rss_end_bytes = 0;
    uint64_t mem_rss_peak_bytes = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes = 0;
    uint64_t mem_phys_peak_bytes = 0;
    
};

// Progress callback type
using tts_progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Streaming decode options. When batch_size > 0, the transformer emits
// audio codes in frame batches that are decoded live via the audio
// decoder's streaming path, and each decoded PCM batch is forwarded to
// `on_pcm`. A final flush drains any trailing partial batch. The
// aggregate PCM is also accumulated into `tts_result::audio` for parity
// with the non-streaming path, but consumers that only care about wire
// bytes can ignore it.
struct streaming_opts {
    int32_t batch_size = 0;
    std::function<bool(const float * pcm, size_t n_samples)> on_pcm;
};

// Main TTS class that orchestrates the full pipeline
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();
    
    // Load all models from directory (auto-detects q8_0 vs f16)
    // model_dir should contain: transformer.gguf, tokenizer.gguf, vocoder.gguf
    bool load_models(const std::string & model_dir);

    // Load models from explicit file paths
    // tts_model_path: path to the TTS GGUF (tokenizer + transformer + encoder)
    // vocoder_model_path: path to the vocoder GGUF (if empty, looks in same directory)
    bool load_model_files(const std::string & tts_model_path,
                          const std::string & vocoder_model_path = "");
    
    // Generate speech from text
    // text: input text to synthesize
    // params: generation parameters
    tts_result synthesize(const std::string & text,
                          const tts_params & params = tts_params());
    
    // Generate speech with voice cloning
    // text: input text to synthesize
    // reference_audio: path to reference audio file (WAV, 24kHz)
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const std::string & reference_audio,
                                      const tts_params & params = tts_params());
    
    // Generate speech with voice cloning from samples
    // text: input text to synthesize
    // ref_samples: reference audio samples (24kHz, mono, normalized to [-1, 1])
    // n_ref_samples: number of reference samples
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const float * ref_samples, int32_t n_ref_samples,
                                      const tts_params & params = tts_params());

    // Extract speaker embedding from reference audio file (without synthesis)
    // reference_audio: path to reference audio file (WAV)
    // embedding: output vector of 1024 float32 values
    bool extract_speaker_embedding(const std::string & reference_audio,
                                    std::vector<float> & embedding);

    // Encode audio to discrete speech codes for ICL voice cloning
    // samples: audio samples (24kHz, mono, normalized to [-1, 1])
    // n_samples: number of samples
    // codes: output vector of codes (n_frames * 16 interleaved)
    // n_frames: output number of frames
    bool encode_speech_codes(const float * samples, int32_t n_samples,
                              std::vector<int32_t> & codes, int32_t & n_frames);

    // Generate speech with pre-extracted speaker embedding
    // text: input text to synthesize
    // embedding: pre-extracted speaker embedding (1024 float32 values)
    // embedding_size: number of elements in embedding (must be 1024)
    // params: generation parameters
    tts_result synthesize_with_embedding(const std::string & text,
                                          const float * embedding, int32_t embedding_size,
                                          const tts_params & params = tts_params(),
                                          const int32_t * ref_codes = nullptr,
                                          int32_t n_ref_frames = 0,
                                          const streaming_opts * stream = nullptr);

    // Streaming overload for non-voice-clone synthesis. See streaming_opts.
    tts_result synthesize(const std::string & text,
                          const tts_params & params,
                          const streaming_opts * stream);

    // Query model info
    int32_t get_hidden_size() const;
    const std::string & get_model_type() const;
    const std::vector<std::string> & get_speaker_names() const;
    const std::vector<int32_t> & get_speaker_ids() const;
    bool has_speaker_encoder() const;

    // Look up speaker token ID by name (-1 if not found)
    int32_t get_speaker_id(const std::string & name) const;

    // Get speaker embedding for a built-in speaker (from codec_embd at speaker token ID)
    bool get_speaker_embedding(const std::string & name, std::vector<float> & embedding);

    // Set progress callback
    void set_progress_callback(tts_progress_callback_t callback);

    // Set abort callback on all loaded component backends (thread-safe).
    // The callback is stored and automatically re-applied after lazy load/reload.
    void set_abort_callback(ggml_abort_callback callback, void * data);

    // Get error message
    const std::string & get_error() const { return error_msg_; }

    // Check if models are loaded
    bool is_loaded() const { return models_loaded_; }
    
private:
    tts_result synthesize_internal(const std::string & text,
                                   const float * speaker_embedding,
                                   const tts_params & params,
                                   tts_result & result,
                                   const int32_t * ref_codes = nullptr,
                                   int32_t n_ref_frames = 0,
                                   const streaming_opts * stream = nullptr);

    bool is_aborted() const { return abort_cb_ && abort_cb_(abort_data_); }
    
    TextTokenizer tokenizer_;
    TTSTransformer transformer_;
    AudioTokenizerEncoder audio_encoder_;
    AudioCodecEncoder codec_encoder_;
    AudioTokenizerDecoder audio_decoder_;
    
    bool models_loaded_ = false;
    bool encoder_loaded_ = false;
    bool codec_encoder_loaded_ = false;
    bool transformer_loaded_ = false;
    bool decoder_loaded_ = false;
    bool low_mem_mode_ = false;
    std::string error_msg_;
    std::string tts_model_path_;
    std::string decoder_model_path_;
    tts_progress_callback_t progress_callback_;
    ggml_abort_callback abort_cb_ = nullptr;
    void * abort_data_ = nullptr;
};

// Utility: Load audio file (WAV format)
bool load_audio_file(const std::string & path, std::vector<float> & samples,
                     int & sample_rate);

// Utility: Save audio file. Format is chosen by the path extension: ".mp3" and
// ".opus"/".ogg" emit compressed audio (when compiled with libav), anything
// else emits 16-bit WAV.
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate);

// Compressed audio output (mp3, opus) via the optional libav/ffmpeg backend.
// Default bitrates in bits/sec; opus is efficient so it targets a lower rate.
enum class audio_codec { mp3, opus };
constexpr int MP3_BITRATE  = 128000;
constexpr int OPUS_BITRATE = 64000;

// true when compressed-audio output was compiled in (libav found at build time)
bool compressed_audio_supported();

// Map a lowercase format or extension name ("mp3", "opus", "ogg") to a codec.
// Returns false for names that are not a compressed audio format.
bool codec_from_name(const std::string & name, audio_codec & out);

// Encode float32 mono samples in [-1,1] to a complete container byte buffer
// (mp3 frames, or ogg/opus). Empty on failure or when unsupported.
std::string encode_compressed(audio_codec codec, const std::vector<float> & samples,
                              int sample_rate);

// Incremental encoder for streaming output. _open returns nullptr when
// unsupported or init fails. Feed chunks with _write, finish with _flush (emits
// trailing bytes), then _close. _write/_flush return container bytes ready for
// the wire (possibly empty for a given chunk while the muxer buffers).
struct compressed_encoder;
compressed_encoder * compressed_encoder_open(audio_codec codec, int sample_rate);
std::string compressed_encoder_write(compressed_encoder * enc, const float * pcm, size_t n);
std::string compressed_encoder_flush(compressed_encoder * enc);
void        compressed_encoder_close(compressed_encoder * enc);

} // namespace qwen3_tts
