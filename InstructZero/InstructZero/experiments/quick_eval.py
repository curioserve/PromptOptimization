import os
import random
import torch
import numpy as np

from automatic_prompt_engineer import data, template, config, evaluate
from data.instruction_induction.load_data import load_data
from misc import get_conf
from run_instructzero import LMForwardAPI


def main():
    # Minimal fast check: single eval() call to verify instruction generation is non-empty
    task = os.environ.get("QCHECK_TASK", "math500_highconf")
    model_dir = os.environ.get("QCHECK_MODEL_DIR", os.environ.get("HF_CACHE_DIR", "/home/tavakoli/prompt_optmization/gpt-oss-20b"))
    model_name = os.environ.get("QCHECK_MODEL_NAME", "gpt-oss-20b")

    random_proj = os.environ.get("QCHECK_RANDOM_PROJ", "uniform")
    intrinsic_dim = int(os.environ.get("QCHECK_INTRINSIC_DIM", "10"))
    n_prompt_tokens = int(os.environ.get("QCHECK_N_PROMPT_TOKENS", "5"))

    print(f"[quick_eval] task={task} model_name={model_name} model_dir={model_dir}")

    induce_data, eval_data = load_data('induce', task), load_data('eval', task)

    # Prepare demos
    prompt_gen_size = min(int(len(induce_data[0])), 100)
    prompt_gen_data, _ = data.create_split(induce_data, prompt_gen_size)
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0] for output in prompt_gen_data[1]]

    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOUTPUT: [OUTPUT]"

    base_conf = '../configs/instruction_induction.yaml'
    conf = get_conf(task, eval_data)

    subsampled_data = data.subsample_data(prompt_gen_data, conf['generation']['num_demos'])
    prompt_gen_template = template.InitQATemplate("[full_DEMO]\n\nThe instruction was to?")
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_prompt = ['\n']
    init_qa = [prompt_gen_template.fill(demos)]

    # Build the forward API
    class Args: pass
    args = Args()
    args.api_model = 'chatgpt'  # use GPT_Forward wrapper for evaluation
    args.alpha = 1.0
    args.sigma = 3.0

    lmf = LMForwardAPI(
        model_name=model_name,
        eval_data=eval_data,
        init_prompt=init_prompt,
        init_qa=init_qa,
        conf=conf,
        base_conf=base_conf,
        prompt_gen_data=prompt_gen_data,
        random_proj=random_proj,
        intrinsic_dim=intrinsic_dim,
        n_prompt_tokens=n_prompt_tokens,
        HF_cache_dir=model_dir,
        args=args,
    )

    # Draw a single latent and evaluate
    z = torch.rand(intrinsic_dim) * 2 - 1  # in [-1, 1]
    print("[quick_eval] calling LMForwardAPI.eval once...")
    dev_perf, instruction_score = lmf.eval(z)
    print("[quick_eval] Done. Dev perf:", dev_perf)
    print("[quick_eval] Instruction:", lmf.return_best_prompt())


if __name__ == "__main__":
    main()
