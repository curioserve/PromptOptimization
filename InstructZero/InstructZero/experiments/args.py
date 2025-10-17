import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--random_proj",
        type=str,
        default="uniform",
        help="The initialization of the projection matrix A."
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=10,
        help="The instrinsic dimension of the projection matrix"
    )
    parser.add_argument(
        "--n_prompt_tokens",
        type=int,
        default=5,
        help="The number of prompt tokens."
    )
    parser.add_argument(
        "--HF_cache_dir",
        type=str,
        default="/data/bobchen/vicuna-13b",
        help="Your vicuna directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed."    
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Set the alpha if the initialization of the projection matrix A is std."    
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=3.0,
        help="Set the beta if the initialization of the projection matrix A is std."    
    )
    parser.add_argument(
        "--api_model",
        type=str,
        default='chatgpt',
        help="The black-box api model. Choices: chatgpt, local_hf"    
    )
    parser.add_argument(
        "--bb_hf_path",
        type=str,
        default=None,
        help="Path to local HF model to use when --api_model local_hf"
    )
    parser.add_argument(
        "--bb_batch_size",
        type=int,
        default=2,
        help="Batch size for local HF black-box evaluator"
    )
    parser.add_argument(
        "--bb_max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens for local HF black-box evaluator"
    )
    parser.add_argument(
        "--bb_torch_dtype",
        type=str,
        default='float16',
        help="torch dtype for local HF black-box evaluator: float16|bfloat16|float32"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='vicuna',
        help="The model name of the open-source LLM."    
    )
    parser.add_argument(
        "--hf_arch",
        type=str,
        default='auto',
        help="When --model_name hf: architecture hint for loading. Options: auto, gpt_neox, llama, mpt"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional: path or HF repo id to load tokenizer from (useful when local tokenizer.json is incompatible)"
    )
    parser.add_argument(
        "--load_in_8bit",
        action='store_true',
        help="Load local HF model in 8-bit (requires bitsandbytes)"
    )
    parser.add_argument(
        "--load_in_4bit",
        action='store_true',
        help="Load local HF model in 4-bit NF4 (requires bitsandbytes)"
    )
    parser.add_argument(
        "--bnb_compute_dtype",
        type=str,
        default='float16',
        help="bitsandbytes compute dtype when using 4-bit: float16 or bfloat16"
    )
    parser.add_argument(
        "--bnb_quant_type",
        type=str,
        default='nf4',
        help="bitsandbytes 4-bit quant type: nf4 or fp4"
    )
    parser.add_argument(
        "--selection",
        type=str,
        default='bo',
        choices=['bo', 'bai'],
        help="Search controller: 'bo' (Bayesian optimization) or 'bai' (fixed-budget BAI).",
    )
    parser.add_argument(
        "--bai_method",
        type=str,
        default='sh',
        choices=['sh', 'cr', 'clst', 'gse'],
        help="BAI strategy when --selection=bai: sequential halving (sh), continuous reject (cr), cluster BAI (clst), or global surrogate elimination (gse).",
    )
    parser.add_argument(
        "--bai_total_budget",
        type=int,
        default=80,
        help="Total number of evaluation calls allocated to the BAI selector.",
    )
    parser.add_argument(
        "--bai_batch_size",
        type=int,
        default=4,
        help="Number of pulls per active arm during a BAI sweep.",
    )
    parser.add_argument(
        "--bai_delta",
        type=float,
        default=0.1,
        help="Confidence level used by continuous reject to prune arms.",
    )
    parser.add_argument(
        "--bai_seed_evals",
        type=int,
        default=24,
        help="Number of Sobol seed evaluations run before the main BAI loop.",
    )
    parser.add_argument(
        "--bai_num_arms",
        type=int,
        default=64,
        help="Number of randomly initialised soft prompts evaluated by the BAI selector.",
    )
    parser.add_argument(
        "--bai_arm_builder",
        type=str,
        default='random',
        choices=['random', 'instruction_first', 'trust_region', 'lhs'],
        help="Method to build candidate arms: random (current), instruction_first, trust_region, or lhs.",
    )
    parser.add_argument(
        "--bai_embedding_model",
        type=str,
        default=None,
        help="Embedding model for CLST/GSE methods (e.g., 'sentence-transformers/all-MiniLM-L6-v2').",
    )
    parser.add_argument(
        "--bai_num_clusters",
        type=int,
        default=8,
        help="Number of clusters for CLST method.",
    )
    parser.add_argument(
        "--bai_surrogate_model",
        type=str,
        default='linear',
        choices=['linear', 'rf'],
        help="Surrogate model for GSE method: linear regression or random forest.",
    )
    args = parser.parse_args()
    return args