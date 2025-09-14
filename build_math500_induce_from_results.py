#!/usr/bin/env python3
import argparse, json, os
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser(description="Build InstructZero induce set from math500 results.")
    ap.add_argument("--results_path", required=True,
                    help="Path to math500_results_20b.json (large JSON with individual_runs).")
    ap.add_argument("--out_task_name", default="math500_highconf",
                    help="Task name to write under raw/induce/{task}.json")
    ap.add_argument("--min_correct", type=int, default=4,
                    help="Minimum number of correct answers (out of total runs) to keep a problem.")
    ap.add_argument("--repo_root", default=os.path.join(os.path.dirname(__file__),
                     "InstructZero", "InstructZero", "experiments"),
                    help="Path to InstructZero/InstructZero/experiments (auto-guess ok if running from repo root).")
    args = ap.parse_args()

    with open(args.results_path, "r") as f:
        data = json.load(f)

    # Aggregate correctness per unique problem
    # key by (problem, ground_truth) to be safe if same text occurs with different answers
    stats = defaultdict(lambda: {"ground_truth": None, "count_correct": 0, "count_total": 0})
    runs = data.get("individual_runs", [])
    for run in runs:
        for r in run.get("results", []):
            prob = r.get("problem", "").strip()
            gt = r.get("ground_truth", "").strip()
            key = (prob, gt)
            stats[key]["ground_truth"] = gt
            stats[key]["count_total"] += 1
            if r.get("is_correct", False):
                stats[key]["count_correct"] += 1

    # Filter by threshold
    selected = [(p, gt, s["count_correct"], s["count_total"])
                for (p, gt), s in stats.items()
                if s["count_correct"] >= args.min_correct]

    selected.sort(key=lambda x: (-x[2], x[0][:64]))  # stable order: most-correct first

    # Build InstructZero induce JSON format: {"metadata": {"num_examples": N}, "examples": {"1": {...}, ...}}
    examples = {}
    for i, (p, gt, c_ok, c_tot) in enumerate(selected, start=1):
        examples[str(i)] = {
            "input": p,
            "output": gt,
            # optional helpful metadata (ignored by loader)
            "meta": {"correct_runs": c_ok, "total_runs": c_tot}
        }

    out = {
        "metadata": {"num_examples": len(examples)},
        "examples": examples
    }

    out_dir = os.path.join(args.repo_root, "data", "instruction_induction", "raw", "induce")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.out_task_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(examples)} examples to {out_path}")

    # Next steps hint
    print("\nNext steps:")
    print(f"  1) Add '{args.out_task_name}' to TASKS in experiments/misc.py")
    print("  2) Also create an execute set (raw/execute/{task}.json) or point eval to another dataset.")
    print("     You can reuse this script to create a quick execute split using --exec-from-induce if desired.")

if __name__ == "__main__":
    main()