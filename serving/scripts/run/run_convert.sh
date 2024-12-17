#!/bin/bash


RUN_DIR=$(dirname "$(readlink -f "$0")")
CORE_DIR=$(dirname "$(dirname "$RUN_DIR")")
SET_DIR="${CORE_DIR}/scripts/set"

source "${SET_DIR}/env/set_vars.sh"
MAMBA_ENV="mgfm_evaluate"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"


LIT_MODEL_DIR="${MODEL_CKPT_DIR}/litgpt"
PTH_MODEL_DIR="${MODEL_CKPT_DIR}/pth"
ST_MODEL_DIR="${MODEL_CKPT_DIR}/safetensors"

CKPT_STEP="step-00078000" # 00078000, 00080000 <-> 00086000
mkdir -p "${PTH_MODEL_DIR}/${CKPT_STEP}"
mkdir -p "${ST_MODEL_DIR}/${CKPT_STEP}"

echo "Convert litgpt to pth ..."
litgpt convert_from_litgpt "${LIT_MODEL_DIR}/${CKPT_STEP}" "${PTH_MODEL_DIR}/${CKPT_STEP}"
cp "${LIT_MODEL_DIR}/${CKPT_STEP}/tokenizer.model" \
    "${LIT_MODEL_DIR}/${CKPT_STEP}/config.json" \
    "${LIT_MODEL_DIR}/${CKPT_STEP}/tokenizer.json" \
    "${PTH_MODEL_DIR}/${CKPT_STEP}"

echo "Convert pth to safetensors ..."
python "${CORE_DIR}/convert/convert_model_pth_to_hf.py" \
   --pth_model_dir "${PTH_MODEL_DIR}" \
   --st_model_dir "${ST_MODEL_DIR}" \
   --model_ckpt "${CKPT_STEP}"
cp "${LIT_MODEL_DIR}/${CKPT_STEP}/tokenizer.model" \
    "${LIT_MODEL_DIR}/${CKPT_STEP}/config.json" \
    "${LIT_MODEL_DIR}/${CKPT_STEP}/tokenizer.json" \
    "${ST_MODEL_DIR}/${CKPT_STEP}"
cp "${LIT_MODEL_DIR}/${CKPT_STEP}/tokenizer.json" \
    "${ST_MODEL_DIR}/${CKPT_STEP}/tokenizer_original.json"

echo "Rebuild safetensors tokenizer ..."
python "${CORE_DIR}/convert/rebuild_model_tokenizer.py" \
    --model_dir "${ST_MODEL_DIR}" \
    --model_ckpt "${CKPT_STEP}"

# https://www.substratus.ai/blog/converting-hf-model-gguf-model/
GGUF_MODEL_DIR="${MODEL_CKPT_DIR}/gguf/gpt2"
# GGUF_MODEL_DIR="${MODEL_CKPT_DIR}/gguf/llama-bpe"
mkdir -p "${GGUF_MODEL_DIR}"

echo "Convert safetensors to gguf ..."
rsync -a --delete "${ST_MODEL_DIR}/${CKPT_STEP}/" "${GGUF_MODEL_DIR}/${CKPT_STEP}/"
python "${CORE_DIR}/convert/llama.cpp/convert_hf_to_gguf.py" "${GGUF_MODEL_DIR}/${CKPT_STEP}" \
    --outtype "f32" \
    --outfile "${GGUF_MODEL_DIR}/${CKPT_STEP}/model.gguf"
rm -f "${GGUF_MODEL_DIR}/${CKPT_STEP}/*.safetensors"
rm -f "${GGUF_MODEL_DIR}/${CKPT_STEP}/model.safetensors.index.json"