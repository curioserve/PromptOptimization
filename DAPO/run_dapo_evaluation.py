#!/usr/bin/env python3
"""
Simple script to run DAPO evaluation with different configurations.
"""

from dapo_evaluator import DAPOEvaluator
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description="Run DAPO Math dataset evaluation")
    parser.add_argument("--model", default="./gpt-oss-20b", help="Model ID to use")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate (None for all)")
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs to perform")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum new tokens to generate")
    parser.add_argument("--output", default="dapo_results.json", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dataset_config", default="all", choices=["all", "en", "cn"], help="Dataset configuration to use")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"Starting DAPO evaluation with:")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.samples}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Dataset config: {args.dataset_config}")
    print(f"  Output: {args.output}")
    
    # Initialize evaluator
    evaluator = DAPOEvaluator(
        model_id=args.model,
        max_new_tokens=args.max_tokens,
        dataset_config=args.dataset_config
    )
    
    try:
        # Run evaluation (single or multiple runs)
        if args.runs == 1:
            summary, results = evaluator.evaluate_single_run(num_samples=args.samples)
            evaluator.all_runs_results = [{'run_number': 1, 'summary': summary, 'results': results}]
            print(f"\nSingle run evaluation complete!")
            print(f"Accuracy: {summary['accuracy']:.3f}")
        else:
            overall_stats = evaluator.evaluate_multiple_runs(num_samples=args.samples, num_runs=args.runs)
            print(f"\nMultiple runs evaluation complete!")
            print(f"Average Accuracy: {overall_stats['accuracy_stats']['mean']:.3f} ± {overall_stats['accuracy_stats']['std']:.3f}")
        
        # Save results
        evaluator.save_results(args.output)
        
        # Print sample results if verbose
        if args.verbose:
            evaluator.print_sample_results()
        
        print(f"Results saved to {args.output}")
        
        print(f"\n✅ Evaluation completed successfully!")
        if args.runs == 1:
            print(f"📊 Accuracy: {summary['accuracy']:.1%}")
        else:
            print(f"📊 Average Accuracy: {overall_stats['accuracy_stats']['mean']:.1%}")
        print(f"📁 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
