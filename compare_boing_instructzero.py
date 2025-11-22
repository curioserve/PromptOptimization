#!/usr/bin/env python3
"""
Comparison script for BOInG vs InstructZero on MATH500 dataset.

This script runs both methods with the same budget and compares results.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, './InstructZero')

from test_boing import (
    setup_gpt2_embeddings,
    create_scoring_function_api,
    BOInG
)

try:
    from InstructZero.InstructZero.experiments.run_instructzero import (
        LMForwardAPI,
        run as run_instructzero
    )
    INSTRUCTZERO_AVAILABLE = True
except ImportError:
    INSTRUCTZERO_AVAILABLE = False
    print("Warning: InstructZero not available. Install dependencies to compare.")


class ComparisonExperiment:
    """Run fair comparison between BOInG and InstructZero."""
    
    def __init__(
        self,
        task: str = "MATH500",
        num_samples: int = 10,
        budget: int = 15,  # Total evaluations
        api_key: str = None,
        use_openrouter: bool = True,
        api_model: str = "meta-llama/llama-3.2-3b-instruct:free",
        output_dir: str = "comparison_results"
    ):
        self.task = task
        self.num_samples = num_samples
        self.budget = budget
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.use_openrouter = use_openrouter
        self.api_model = api_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            'experiment_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'config': {
                'task': task,
                'num_samples': num_samples,
                'budget': budget,
                'api_model': api_model,
                'use_openrouter': use_openrouter
            },
            'boing': {},
            'instructzero': {}
        }
    
    def run_boing(
        self,
        init_points: int = 5,
        iterations: int = 10,
        n_tokens: int = 5,
        latent_dim: int = 50
    ) -> Dict[str, Any]:
        """Run BOInG optimization."""
        print("\n" + "="*60)
        print("Running BOInG")
        print("="*60)
        
        start_time = time.time()
        
        # Setup
        embeddings, tokens, _ = setup_gpt2_embeddings()
        scoring_func = create_scoring_function_api(
            api_key=self.api_key,
            model_name=self.api_model,
            num_samples=self.num_samples,
            use_openrouter=self.use_openrouter
        )
        
        # Initialize BOInG
        boing = BOInG(
            scoring_func=scoring_func,
            embedding_matrix=embeddings,
            token_list=tokens,
            n_tokens=n_tokens,
            latent_dim=latent_dim,
            beta=2.0,
            lambda_penalty=10.0
        )
        
        # Optimize
        best_instruction, best_score = boing.optimize(
            iterations=iterations,
            init_points=init_points,
            verbose=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Collect results
        results = {
            'best_instruction': best_instruction,
            'best_score': float(best_score),
            'total_time': elapsed_time,
            'num_evaluations': init_points + iterations,
            'convergence_history': {
                'scores': [float(score) for score in boing.y_train],
                'instructions': [str(instr) for instr in boing.y_train]
            },
            'hyperparameters': {
                'n_tokens': n_tokens,
                'latent_dim': latent_dim,
                'init_points': init_points,
                'iterations': iterations
            }
        }
        
        print(f"\nBOInG Results:")
        print(f"  Best Instruction: {best_instruction}")
        print(f"  Best Score: {best_score:.4f}")
        print(f"  Total Time: {elapsed_time:.2f}s")
        
        return results
    
    def run_instructzero(
        self,
        intrinsic_dim: int = 10,
        n_prompt_tokens: int = 5,
        hf_model_path: str = None
    ) -> Dict[str, Any]:
        """Run InstructZero optimization."""
        if not INSTRUCTZERO_AVAILABLE:
            raise ImportError("InstructZero not available. Install dependencies.")
        
        print("\n" + "="*60)
        print("Running InstructZero")
        print("="*60)
        
        start_time = time.time()
        
        # Note: InstructZero requires more setup
        # This is a simplified version - adjust based on your setup
        print("Warning: InstructZero requires open-source LLM setup.")
        print("Please configure InstructZero separately.")
        
        # Placeholder - implement based on your InstructZero setup
        results = {
            'best_instruction': "Not implemented - requires LLM setup",
            'best_score': 0.0,
            'total_time': 0.0,
            'num_evaluations': self.budget,
            'hyperparameters': {
                'intrinsic_dim': intrinsic_dim,
                'n_prompt_tokens': n_prompt_tokens
            }
        }
        
        return results
    
    def compare_results(self, boing_results: Dict, iz_results: Dict):
        """Compare and analyze results."""
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        comparison = {
            'performance': {
                'boing': {
                    'best_score': boing_results['best_score'],
                    'avg_score': np.mean(boing_results['convergence_history']['scores']) if 'convergence_history' in boing_results else boing_results['best_score']
                },
                'instructzero': {
                    'best_score': iz_results['best_score'],
                    'avg_score': iz_results['best_score']  # Placeholder
                }
            },
            'efficiency': {
                'boing': {
                    'time_per_eval': boing_results['total_time'] / boing_results['num_evaluations'],
                    'total_time': boing_results['total_time']
                },
                'instructzero': {
                    'time_per_eval': iz_results['total_time'] / iz_results['num_evaluations'] if iz_results['num_evaluations'] > 0 else 0,
                    'total_time': iz_results['total_time']
                }
            }
        }
        
        # Print comparison
        print(f"\nPerformance:")
        print(f"  BOInG Best Score:      {comparison['performance']['boing']['best_score']:.4f}")
        print(f"  InstructZero Best:     {comparison['performance']['instructzero']['best_score']:.4f}")
        print(f"  Difference:            {comparison['performance']['instructzero']['best_score'] - comparison['performance']['boing']['best_score']:.4f}")
        
        print(f"\nEfficiency:")
        print(f"  BOInG Time:            {comparison['efficiency']['boing']['total_time']:.2f}s")
        print(f"  InstructZero Time:     {comparison['efficiency']['instructzero']['total_time']:.2f}s")
        print(f"  Speedup:               {comparison['efficiency']['instructzero']['total_time'] / comparison['efficiency']['boing']['total_time']:.2f}x" if comparison['efficiency']['boing']['total_time'] > 0 else "N/A")
        
        return comparison
    
    def generate_report(self, boing_results: Dict, iz_results: Dict, comparison: Dict):
        """Generate comparison report."""
        report = {
            'experiment_metadata': self.results['experiment_id'],
            'config': self.results['config'],
            'boing_results': boing_results,
            'instructzero_results': iz_results,
            'comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save JSON report
        report_file = self.output_dir / f"comparison_{self.results['experiment_id']}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save text summary
        summary_file = self.output_dir / f"summary_{self.results['experiment_id']}.txt"
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("BOInG vs InstructZero Comparison Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Experiment ID: {self.results['experiment_id']}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Budget: {self.budget} evaluations\n")
            f.write(f"API Model: {self.api_model}\n\n")
            
            f.write("BOInG Results:\n")
            f.write(f"  Best Instruction: {boing_results['best_instruction']}\n")
            f.write(f"  Best Score: {boing_results['best_score']:.4f}\n")
            f.write(f"  Total Time: {boing_results['total_time']:.2f}s\n\n")
            
            f.write("InstructZero Results:\n")
            f.write(f"  Best Instruction: {iz_results['best_instruction']}\n")
            f.write(f"  Best Score: {iz_results['best_score']:.4f}\n")
            f.write(f"  Total Time: {iz_results['total_time']:.2f}s\n\n")
            
            f.write("Comparison:\n")
            f.write(f"  Performance Difference: {comparison['performance']['instructzero']['best_score'] - comparison['performance']['boing']['best_score']:.4f}\n")
            f.write(f"  Time Ratio: {comparison['efficiency']['instructzero']['total_time'] / comparison['efficiency']['boing']['total_time']:.2f}x\n" if comparison['efficiency']['boing']['total_time'] > 0 else "  Time Ratio: N/A\n")
        
        print(f"\nReport saved to: {report_file}")
        print(f"Summary saved to: {summary_file}")
        
        return report
    
    def run_comparison(self):
        """Run complete comparison experiment."""
        print("="*60)
        print("BOInG vs InstructZero Comparison")
        print("="*60)
        print(f"Task: {self.task}")
        print(f"Budget: {self.budget} evaluations")
        print(f"API Model: {self.api_model}")
        print("="*60)
        
        # Run BOInG
        boing_results = self.run_boing(
            init_points=5,
            iterations=10
        )
        self.results['boing'] = boing_results
        
        # Run InstructZero (if available)
        if INSTRUCTZERO_AVAILABLE:
            iz_results = self.run_instructzero(
                intrinsic_dim=10,
                n_prompt_tokens=5
            )
            self.results['instructzero'] = iz_results
        else:
            print("\nSkipping InstructZero (not available)")
            iz_results = {
                'best_instruction': "N/A",
                'best_score': 0.0,
                'total_time': 0.0,
                'num_evaluations': 0
            }
        
        # Compare
        comparison = self.compare_results(boing_results, iz_results)
        
        # Generate report
        report = self.generate_report(boing_results, iz_results, comparison)
        
        return report


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare BOInG vs InstructZero")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--use_openrouter", action="store_true", help="Use OpenRouter")
    parser.add_argument("--api_model", type=str, default="meta-llama/llama-3.2-3b-instruct:free", help="Model to use")
    parser.add_argument("--samples", type=int, default=10, help="Number of evaluation samples")
    parser.add_argument("--budget", type=int, default=15, help="Total evaluation budget")
    parser.add_argument("--output", type=str, default="comparison_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Run comparison
    experiment = ComparisonExperiment(
        num_samples=args.samples,
        budget=args.budget,
        api_key=args.api_key,
        use_openrouter=args.use_openrouter,
        api_model=args.api_model,
        output_dir=args.output
    )
    
    report = experiment.run_comparison()
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)


if __name__ == "__main__":
    main()


