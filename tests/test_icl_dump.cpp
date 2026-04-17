// dumps reference codes + speaker embedding for a wav, for diffing against
// /tmp/python_ref_codes.npy + /tmp/python_spk_emb.npy

#include "qwen3_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --tts <gguf> --vocoder <gguf> --audio <wav>\n", prog);
}

int main(int argc, char ** argv) {
    std::string tts_path;
    std::string vocoder_path;
    std::string audio_path;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tts") == 0 && i + 1 < argc) {
            tts_path = argv[++i];
        } else if (strcmp(argv[i], "--vocoder") == 0 && i + 1 < argc) {
            vocoder_path = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (tts_path.empty() || vocoder_path.empty() || audio_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    qwen3_tts::Qwen3TTS tts;
    if (!tts.load_model_files(tts_path, vocoder_path)) {
        fprintf(stderr, "load_model_files failed: %s\n", tts.get_error().c_str());
        return 1;
    }

    std::vector<float> samples;
    int sr = 0;
    if (!qwen3_tts::load_audio_file(audio_path, samples, sr)) {
        fprintf(stderr, "load_audio_file failed: %s\n", audio_path.c_str());
        return 1;
    }
    fprintf(stderr, "loaded wav: %zu samples @ %d Hz\n", samples.size(), sr);

    // resample to 24k using the same linear path the production code uses
    if (sr != 24000) {
        double ratio = (double)sr / 24000.0;
        int out_len = (int)((double)samples.size() / ratio);
        std::vector<float> resampled(out_len);
        for (int i = 0; i < out_len; ++i) {
            double src = i * ratio;
            int i0 = (int)src;
            int i1 = i0 + 1;
            double f = src - i0;
            if (i1 >= (int)samples.size()) {
                resampled[i] = samples.back();
            } else {
                resampled[i] = (float)((1.0 - f) * samples[i0] + f * samples[i1]);
            }
        }
        samples = std::move(resampled);
        fprintf(stderr, "resampled to 24kHz: %zu samples\n", samples.size());
    }

    std::vector<float> emb;
    if (!tts.extract_speaker_embedding(audio_path, emb)) {
        fprintf(stderr, "extract_speaker_embedding failed: %s\n", tts.get_error().c_str());
        return 1;
    }
    double norm = 0.0;
    for (float v : emb) norm += (double)v * v;
    norm = std::sqrt(norm);

    printf("=== speaker embedding ===\n");
    printf("size: %zu\n", emb.size());
    printf("norm: %.4f\n", norm);
    printf("first8:");
    for (int i = 0; i < 8 && i < (int)emb.size(); ++i) printf(" %.4f", emb[i]);
    printf("\n");
    printf("last8:");
    for (int i = std::max(0, (int)emb.size() - 8); i < (int)emb.size(); ++i) printf(" %.4f", emb[i]);
    printf("\n");

    std::vector<int32_t> codes;
    int32_t n_frames = 0;
    if (!tts.encode_speech_codes(samples.data(), (int32_t)samples.size(), codes, n_frames)) {
        fprintf(stderr, "encode_speech_codes failed: %s\n", tts.get_error().c_str());
        return 1;
    }
    const int n_cb = 16;

    printf("\n=== ref codes ===\n");
    printf("n_frames: %d, n_codebooks: %d\n", n_frames, n_cb);
    printf("first frame:");
    for (int cb = 0; cb < n_cb; ++cb) printf(" %d", codes[0 * n_cb + cb]);
    printf("\n");
    printf("last frame [%d]:", n_frames - 1);
    for (int cb = 0; cb < n_cb; ++cb) printf(" %d", codes[(n_frames - 1) * n_cb + cb]);
    printf("\n");

    // write codes as raw int32 to /tmp for python diff
    FILE * fc = fopen("/tmp/cpp_ref_codes.bin", "wb");
    if (fc) {
        fwrite(codes.data(), sizeof(int32_t), codes.size(), fc);
        fclose(fc);
        printf("\nwrote %zu ints to /tmp/cpp_ref_codes.bin\n", codes.size());
    }
    FILE * fe = fopen("/tmp/cpp_spk_emb.bin", "wb");
    if (fe) {
        fwrite(emb.data(), sizeof(float), emb.size(), fe);
        fclose(fe);
        printf("wrote %zu floats to /tmp/cpp_spk_emb.bin\n", emb.size());
    }

    return 0;
}
