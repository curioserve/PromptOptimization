#!/usr/bin/env python3
import argparse, json, os
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser(description="Build InstructZero induce and execute sets from MATH500 results.")
    ap.add_argument("--results_path", required=True,
                    help="Path to results JSON (e.g., Results/math500_results_20b.json) containing individual_runs.")
    ap.add_argument("--out_task_name", default="math500_highconf_COT",
                    help="Base task name to write as raw/induce/{task}.json and raw/execute/{task}.json")
    ap.add_argument("--min_correct", type=int, default=4,
                    help="Threshold: induce includes problems with correct >= min_correct; execute includes the rest.")
    ap.add_argument("--repo_root", default=os.path.join(os.path.dirname(__file__),
                     "InstructZero", "InstructZero", "experiments"),
                    help="Path to InstructZero/InstructZero/experiments (auto-guess ok if running from repo root).")
    args = ap.parse_args()

    with open(args.results_path, "r") as f:
        data = json.load(f)

    # Aggregate correctness per unique problem
    # key by (problem, ground_truth) to be safe if same text occurs with different answers
    stats = defaultdict(lambda: {"ground_truth": None, "full_reasoning": None, "count_correct": 0, "count_total": 0})
    runs = data.get("individual_runs", [])
    for run in runs:
        for r in run.get("results", []):
            prob = r.get("problem", "").strip()
            gt = r.get("ground_truth", "").strip()
            # Get full reasoning from LLM (prefer reasoning_chain, fallback to generated_text)
            full_reasoning = r.get("reasoning_chain", "") or r.get("generated_text", "")
            key = (prob, gt)
            stats[key]["ground_truth"] = gt
            stats[key]["count_total"] += 1
            # Store the full reasoning from a correct answer (if available)
            if r.get("is_correct", False):
                stats[key]["count_correct"] += 1
                if full_reasoning and not stats[key]["full_reasoning"]:
                    stats[key]["full_reasoning"] = full_reasoning.strip()

    # Split by threshold - include full reasoning for high confidence examples
    high_conf = [(p, gt, s["count_correct"], s["count_total"], s["full_reasoning"]) 
                 for (p, gt), s in stats.items() if s["count_correct"] >= args.min_correct]
    low_conf  = [(p, gt, s["count_correct"], s["count_total"], s["full_reasoning"]) 
                 for (p, gt), s in stats.items() if s["count_correct"] <  args.min_correct]

    # Stable sorting for reproducibility
    high_conf.sort(key=lambda x: (-x[2], x[0][:64]))
    low_conf.sort(key=lambda x: (-x[2], x[0][:64]))

    def to_iz_json(rows, use_full_reasoning=False):
        examples = {}
        for i, (p, gt, c_ok, c_tot, full_reasoning) in enumerate(rows, start=1):
            # For COT tasks, use full reasoning as output; otherwise use ground truth
            if use_full_reasoning and full_reasoning:
                output = full_reasoning
            else:
                output = gt
            
            examples[str(i)] = {
                "input": p,
                "output": output,
                "meta": {
                    "correct_runs": c_ok, 
                    "total_runs": c_tot,
                    "has_full_reasoning": bool(full_reasoning)
                }
            }
        return {"metadata": {"num_examples": len(examples)}, "examples": examples}

    # For COT tasks, use full reasoning as output
    use_cot = "COT" in args.out_task_name.upper()
    induce_json = to_iz_json(high_conf, use_full_reasoning=use_cot)
    execute_json = to_iz_json(low_conf, use_full_reasoning=use_cot)

    # Write induce
    induce_dir = os.path.join(args.repo_root, "data", "instruction_induction", "raw", "induce")
    os.makedirs(induce_dir, exist_ok=True)
    induce_path = os.path.join(induce_dir, f"{args.out_task_name}.json")
    with open(induce_path, "w") as f:
        json.dump(induce_json, f, ensure_ascii=False, indent=2)

    # Write execute
    execute_dir = os.path.join(args.repo_root, "data", "instruction_induction", "raw", "execute")
    os.makedirs(execute_dir, exist_ok=True)
    execute_path = os.path.join(execute_dir, f"{args.out_task_name}.json")
    with open(execute_path, "w") as f:
        json.dump(execute_json, f, ensure_ascii=False, indent=2)

    # Count examples with full reasoning
    high_conf_with_reasoning = sum(1 for _, _, _, _, reasoning in high_conf if reasoning)
    low_conf_with_reasoning = sum(1 for _, _, _, _, reasoning in low_conf if reasoning)
    
    print("Done.")
    print(f"  Task type: {'COT (Chain of Thought)' if use_cot else 'Standard (Ground Truth)'}")
    print(f"  Induce:  {len(high_conf)} examples -> {induce_path}")
    print(f"    - With full reasoning: {high_conf_with_reasoning}/{len(high_conf)}")
    print(f"  Execute: {len(low_conf)} examples  -> {execute_path}")
    print(f"    - With full reasoning: {low_conf_with_reasoning}/{len(low_conf)}")
    
    if use_cot:
        print(f"  COT mode: Using full LLM reasoning as output instead of ground truth answers")

if __name__ == "__main__":
    main()