#!/usr/bin/env python3
"""Download the MathArena HMMT February 2025 dataset from HuggingFace and
convert it into the InstructZero JSON format for both induction (training)
and execution (evaluation) splits.

Usage
-----
python build_hmmt_feb_2025_dataset.py --repo_root /path/to/PromptOptimization/InstructZero/InstructZero/experiments

If run from the repository root, no arguments are needed.
"""
import argparse
import json
import os
import random
from typing import List

from datasets import load_dataset

TASK_NAME = "hmmt_feb_2025"
HF_DATASET = "MathArena/hmmt_feb_2025"


def build_examples() -> List[dict]:
    ds = load_dataset(HF_DATASET, split="train")
    examples = []
    for row in ds:
        examples.append({
            "input": row["problem"].strip(),
            "output": row["answer"].strip()
        })
    return examples


def main():
    print("Building InstructZero datasets from HMMT Feb 2025 competition.")
    ap = argparse.ArgumentParser(description="Build InstructZero datasets from HMMT Feb 2025 competition.")
    ap.add_argument("--repo_root", default=os.path.join(os.path.dirname(__file__),
                                                         "InstructZero", "InstructZero", "experiments"),
                    help="Path to InstructZero/InstructZero/experiments directory.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/eval split.")
    ap.add_argument("--eval_fraction", type=float, default=0.2,
                    help="Fraction of problems to put into the execute (evaluation) split.")
    args = ap.parse_args()

    random.seed(args.seed)

    examples = build_examples()
    random.shuffle(examples)

    split_idx = int(len(examples) * (1 - args.eval_fraction))
    induce_examples = examples[:split_idx]
    exec_examples = examples[split_idx:]

    def to_json_dict(ex):
        return {
            "metadata": {"num_examples": len(ex)},
            "examples": {str(i + 1): ex_i for i, ex_i in enumerate(ex)}
        }

    out_dir_base = os.path.join(args.repo_root, "data", "instruction_induction", "raw")
    induce_path = os.path.join(out_dir_base, "induce", f"{TASK_NAME}.json")
    exec_path = os.path.join(out_dir_base, "execute", f"{TASK_NAME}.json")

    os.makedirs(os.path.dirname(induce_path), exist_ok=True)
    os.makedirs(os.path.dirname(exec_path), exist_ok=True)

    with open(induce_path, "w") as f:
        json.dump(to_json_dict(induce_examples), f, ensure_ascii=False, indent=2)
    with open(exec_path, "w") as f:
        json.dump(to_json_dict(exec_examples), f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(induce_examples)} induce examples to {induce_path}")
    print(f"Wrote {len(exec_examples)} execute examples to {exec_path}")

    print("\nNext steps:")
    print(f"  - Ensure '{TASK_NAME}' is listed in TASKS inside experiments/misc.py (already added if you edited it).")
    print("  - Run InstructZero with the new task, e.g.: python run_instructzero.py --task hmmt_feb_2025")


if __name__ == "__main__":
    print("Building InstructZero datasets from HMMT Feb 2025 competition.")
    main()
