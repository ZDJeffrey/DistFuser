set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Select the model type
export MODEL_TYPE="fast"
# Configuration for different model types
# model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["full"]="/file_system/models/dit/HiDream-I1-Full 50"
    ["dev"]="/file_system/models/dit/HiDream-I1-Dev 28"
    ["fast"]="/file_system/models/dit/HiDream-I1-Fast 16"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

LLAMA_MODEL="/file_system/models/Meta-Llama/Meta-Llama-3.1-8B-Instruct"

# task args
TASK_ARGS="--height 1024 --width 1024 --guidance_scale 0.0 --seed 42"

# On 2 gpus, ulysses=2, ring=1, cfg_parallel=1
PARALLEL_ARGS="--dit_num_gpus 2 --ulysses_degree 2 --ring_degree 1"
# CFG_ARGS="--use_cfg_parallel"

# Merge textencoder and vae in one resource pool
# MERGE_ARGS="--merge_textencoder_vae"

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

python ./examples/worker/hidream_example.py \
--model $MODEL_ID \
--llama_model $LLAMA_MODEL \
--model_type $MODEL_TYPE \
$PARALLEL_ARGS \
$CFG_ARGS \
$TASK_ARGS \
$MERGE_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 2 \
--prompt "A beautiful landscape with mountains and a river" \
--negative_prompt "blurry, low quality, bad lighting"