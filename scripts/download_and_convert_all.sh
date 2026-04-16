#!/bin/bash
# download all qwen3-tts model variants from huggingface and convert to GGUF.
#
# produces F16 and Q8_0 quants for each TTS model variant, plus the
# shared tokenizer/vocoder. output is organized into HF-ready upload
# directories under models/gguf/<ModelName>-GGUF/.
#
# requirements: hf cli, python3 with torch, safetensors, gguf, tqdm
#
# usage:
#   ./scripts/download_and_convert_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${REPO_DIR}/models"
SAFETENSOR_DIR="${MODELS_DIR}/safetensor"
GGUF_DIR="${MODELS_DIR}/gguf"

TTS_MODELS=(
    "Qwen3-TTS-12Hz-0.6B-Base"
    "Qwen3-TTS-12Hz-0.6B-CustomVoice"
    "Qwen3-TTS-12Hz-1.7B-Base"
    "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)
TOKENIZER_MODEL="Qwen3-TTS-Tokenizer-12Hz"
ALL_MODELS=("${TTS_MODELS[@]}" "$TOKENIZER_MODEL")

TTS_QUANTS=("f16" "q8_0")
TOKENIZER_QUANTS=("f16")

# activate the project venv if it exists
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
fi

mkdir -p "$SAFETENSOR_DIR" "$GGUF_DIR"

echo "=== downloading models ==="

for model in "${ALL_MODELS[@]}"; do
    dest="$SAFETENSOR_DIR/$model"
    if [ -d "$dest" ] && [ -n "$(ls "$dest"/*.safetensors 2>/dev/null || ls "$dest"/speech_tokenizer/*.safetensors 2>/dev/null)" ]; then
        echo "  $model: already downloaded"
    else
        echo "  downloading Qwen/$model ..."
        hf download "Qwen/$model" --local-dir "$dest"
    fi
done

echo ""
echo "=== converting TTS models ==="

for model in "${TTS_MODELS[@]}"; do
    outdir="$GGUF_DIR/${model}-GGUF"
    mkdir -p "$outdir"

    for quant in "${TTS_QUANTS[@]}"; do
        quant_upper=$(echo "$quant" | tr '[:lower:]' '[:upper:]')
        outfile="$outdir/${model}-${quant_upper}.gguf"

        if [ -f "$outfile" ]; then
            echo "  $outfile: already exists, skipping"
            continue
        fi

        echo "  converting $model ($quant) -> $outfile"
        python3 "$SCRIPT_DIR/convert_tts_to_gguf.py" \
            --input "$SAFETENSOR_DIR/$model" \
            --output "$outfile" \
            --type "$quant"
    done
done

echo ""
echo "=== converting tokenizer/vocoder ==="

outdir="$GGUF_DIR/${TOKENIZER_MODEL}-GGUF"
mkdir -p "$outdir"

for quant in "${TOKENIZER_QUANTS[@]}"; do
    quant_upper=$(echo "$quant" | tr '[:lower:]' '[:upper:]')
    outfile="$outdir/${TOKENIZER_MODEL}-${quant_upper}.gguf"

    if [ -f "$outfile" ]; then
        echo "  $outfile: already exists, skipping"
        continue
    fi

    echo "  converting $TOKENIZER_MODEL ($quant) -> $outfile"
    python3 "$SCRIPT_DIR/convert_tokenizer_to_gguf.py" \
        --input "$SAFETENSOR_DIR/$TOKENIZER_MODEL" \
        --output "$outfile" \
        --type "$quant"
done

echo ""
echo "=== preparing READMEs for HF upload ==="

for model in "${ALL_MODELS[@]}"; do
    src="$SAFETENSOR_DIR/$model/README.md"
    dst="$GGUF_DIR/${model}-GGUF/README.md"

    if [ ! -f "$src" ]; then
        echo "  warning: no README for $model, skipping"
        continue
    fi

    if [ -f "$dst" ]; then
        echo "  $model: README already exists"
        continue
    fi

    # insert GGUF note after frontmatter
    awk -v model="$model" '
        BEGIN { fm=0; done=0 }
        /^---$/ && !done {
            print
            fm++
            if (fm == 2) {
                print ""
                print "> **GGUF quantizations** for use with [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) (and forks)."
                print "> Converted from [Qwen/" model "](https://huggingface.co/Qwen/" model ") using `scripts/convert_tts_to_gguf.py`."
                print "> Available quants: F16, Q8_0."
                done=1
            }
            next
        }
        { print }
    ' "$src" > "$dst"
    echo "  wrote $dst"
done

echo ""
echo "=== done ==="
echo ""
echo "upload directories ready in $GGUF_DIR:"
for d in "$GGUF_DIR"/*/; do
    echo "  $(basename "$d")/"
    ls -1h "$d" | sed 's/^/    /'
done
