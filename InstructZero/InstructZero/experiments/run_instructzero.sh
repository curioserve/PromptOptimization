export CUDA_VISIBLE_DEVICES=0
# Unbuffered python output for real-time logs
export PYTHONUNBUFFERED=1
SFT=5
RANDOM_PROJ='uniform'
INTRINSIC_DIM=10
# model_dir='lmsys/vicuna-13b-v1.3'
# MODEL_NAME='vicuna'
model_dir='/home/tavakoli/prompt_optmization/gpt-oss-20b'
MODEL_NAME='gpt-oss-20b'
# Prefer HF_HOME (TRANSFORMERS_CACHE is deprecated in v5)
export HF_HOME=${HF_HOME:-/fs/nexus-scratch/bobchen}
export TRANSFORMERS_CACHE=$HF_HOME
# Optionally source a local secrets file (not committed) for OpenRouter/OpenAI keys
# Try repo root, current working directory, then $HOME
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [ -f "$REPO_ROOT/.openrouter.env" ]; then source "$REPO_ROOT/.openrouter.env"; fi
if [ -f "$PWD/.openrouter.env" ]; then source "$PWD/.openrouter.env"; fi
if [ -f "$HOME/.openrouter.env" ]; then source "$HOME/.openrouter.env"; fi
# Configure evaluation black-box via OpenRouter
export OPENAI_API_BASE=${OPENAI_API_BASE:-https://openrouter.ai/api/v1}
# If OPENAI_API_KEY is not set, but OPENROUTER_API_KEY is, use that
if [ -z "$OPENAI_API_KEY" ] && [ -n "$OPENROUTER_API_KEY" ]; then export OPENAI_API_KEY="$OPENROUTER_API_KEY"; fi
export EVAL_API_MODEL=${EVAL_API_MODEL:-openai/gpt-oss-20b}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-10}
# Timeout for OpenAI/OpenRouter requests (seconds)
export OPENAI_TIMEOUT=${OPENAI_TIMEOUT:-60}

datasets=(informal_to_formal odd_one_out second_word_letter synonyms word_sorting letters_list)


python experiments/run_instructzero.py \
--task math500_highconf \
--random_proj ${RANDOM_PROJ} \
--n_prompt_tokens $SFT \
--intrinsic_dim $INTRINSIC_DIM \
--HF_cache_dir ${model_dir} \
--model_name ${MODEL_NAME}
