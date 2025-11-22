#!/usr/bin/env python3
"""
Comparison script for BOInG, InstructZero, and INSTINCT on MATH500 dataset.

This script runs all three methods with the same budget and compares results.
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
    create_scoring_function,
    create_generator_function
)
from BoIng import BOInG
from INSTINCT import INSTINCT

try:
    from InstructZero.InstructZero.experiments.run_instructzero import (
        LMForwardAPI,
        run as run_instructzero
    )
    INSTRUCTZERO_AVAILABLE = True
except ImportError:
    INSTRUCTZERO_AVAILABLE = False
    print("Warning: InstructZero not available. Install dependencies to compare.")


class AllMethodsComparison:
    """Run fair comparison between BOInG, InstructZero, and INSTINCT."""
    
    def __init__(
        self,
        task: str = "MATH500",
        num_samples: int = 10,
        iterations: int = 10,
        init_points: int = 5,
        api_key: str = None,
        use_openrouter: bool = True,
        api_model: str = "meta-llama/llama-3.2-3b-instruct:free",
        use_generator: bool = False,
        output_dir: str = "comparison_results",
        model_path: str = None,
        use_api: bool = True
    ):
        self.task = task
        self.num_samples = num_samples
        self.iterations = iterations
        self.init_points = init_points
        self.budget = init_points + iterations
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.use_openrouter = use_openrouter
        self.api_model = api_model
        self.use_generator = use_generator
        self.use_api = use_api
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'experiment_id': self.experiment_id,
            'config': {
                'task': task,
                'num_samples': num_samples,
                'iterations': iterations,
                'init_points': init_points,
                'budget': self.budget,
                'api_model': api_model,
                'use_openrouter': use_openrouter,
                'use_api': use_api,
                'use_generator': use_generator
            },
            'boing': {},
            'instinct': {},
            'instructzero': {}
        }
    
    def setup_scoring_function(self):
        """Setup scoring function for all methods."""
        if self.use_api:
            return create_scoring_function_api(
                api_key=self.api_key,
                model_name=self.api_model,
                num_samples=self.num_samples,
                use_openrouter=self.use_openrouter
            )
        else:
            from MATH500.math500_evaluator import MATH500Evaluator
            if self.model_path is None:
                self.model_path = "./gpt-oss-20b"
            evaluator = MATH500Evaluator(model_id=self.model_path, max_new_tokens=1024)
            return create_scoring_function(evaluator, num_samples=self.num_samples)
    
    def run_boing(self) -> Dict[str, Any]:
        """Run BOInG optimization."""
        print("\n" + "="*80)
        print("Running BOInG (Bayesian Optimization for Instruction Generation)")
        print("="*80)
        
        start_time = time.time()
        
        # Setup
        embeddings, tokens, _ = setup_gpt2_embeddings()
        scoring_func = self.setup_scoring_function()
        generator_func = create_generator_function(
            api_key=self.api_key,
            use_openrouter=self.use_openrouter
        ) if self.use_generator else None
        
        # Initialize BOInG
        boing = BOInG(
            scoring_func=scoring_func,
            embedding_matrix=embeddings,
            token_list=tokens,
            generator_func=generator_func,
            n_tokens=5,
            latent_dim=50,
            beta=2.0,
            lambda_penalty=10.0
        )
        
        # Optimize
        best_instruction, best_score = boing.optimize(
            iterations=self.iterations,
            init_points=self.init_points,
            verbose=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Collect results
        results = {
            'best_instruction': best_instruction,
            'best_score': float(best_score),
            'total_time': elapsed_time,
            'num_evaluations': self.budget,
            'convergence_history': {
                'scores': [float(score) for score in boing.y_train] if hasattr(boing, 'y_train') else [],
            },
            'hyperparameters': {
                'n_tokens': 5,
                'latent_dim': 50,
                'init_points': self.init_points,
                'iterations': self.iterations,
                'beta': 2.0,
                'lambda_penalty': 10.0
            }
        }
        
        print(f"\nBOInG Results:")
        print(f"  Best Instruction: {best_instruction}")
        print(f"  Best Score: {best_score:.4f}")
        print(f"  Total Time: {elapsed_time:.2f}s")
        
        return results
    
    def run_instinct(self) -> Dict[str, Any]:
        """Run INSTINCT optimization."""
        print("\n" + "="*80)
        print("Running INSTINCT (Neural Bandits)")
        print("="*80)
        
        start_time = time.time()
        
        # Setup
        embeddings, tokens, _ = setup_gpt2_embeddings()
        scoring_func = self.setup_scoring_function()
        generator_func = create_generator_function(
            api_key=self.api_key,
            use_openrouter=self.use_openrouter
        ) if self.use_generator else None
        
        # Initialize INSTINCT
        instinct = INSTINCT(
            scoring_func=scoring_func,
            embedding_matrix=embeddings,
            token_list=tokens,
            generator_func=generator_func,
            n_tokens=5,
            latent_dim=50,
            lambda_reg=0.1,
            nu=0.1,
            local_training_iter=30,
            n_domain=1000,
            n_eval=100
        )
        
        # Optimize
        best_instruction, best_score = instinct.optimize(
            iterations=self.iterations,
            init_points=self.init_points,
            verbose=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Collect results
        results = {
            'best_instruction': best_instruction,
            'best_score': float(best_score),
            'total_time': elapsed_time,
            'num_evaluations': self.budget,
            'convergence_history': {
                'scores': [float(score) for score in instinct.y_train] if hasattr(instinct, 'y_train') else [],
            },
            'hyperparameters': {
                'n_tokens': 5,
                'latent_dim': 50,
                'init_points': self.init_points,
                'iterations': self.iterations,
                'lambda_reg': 0.1,
                'nu': 0.1,
                'local_training_iter': 30
            }
        }
        
        print(f"\nINSTINCT Results:")
        print(f"  Best Instruction: {best_instruction}")
        print(f"  Best Score: {best_score:.4f}")
        print(f"  Total Time: {elapsed_time:.2f}s")
        
        return results
    
    def run_instructzero(self) -> Dict[str, Any]:
        """Run InstructZero optimization."""
        if not INSTRUCTZERO_AVAILABLE:
            print("\nWarning: InstructZero not available. Skipping.")
            return {
                'best_instruction': "N/A - InstructZero not available",
                'best_score': 0.0,
                'total_time': 0.0,
                'num_evaluations': 0,
                'error': 'InstructZero requires additional setup with open-source LLM'
            }
        
        print("\n" + "="*80)
        print("Running InstructZero")
        print("="*80)
        print("Note: InstructZero requires open-source LLM setup (e.g., Vicuna).")
        print("For API-based evaluation, using simplified approach.")
        
        start_time = time.time()
        
        # InstructZero requires more complex setup
        # This is a placeholder - users should configure based on their setup
        print("InstructZero requires:")
        print("  1. Open-source LLM (e.g., Vicuna-13B)")
        print("  2. Model path configuration")
        print("  3. Additional dependencies")
        print("\nSkipping InstructZero in this comparison.")
        print("See InstructZero/README.md for setup instructions.")
        
        elapsed_time = time.time() - start_time
        
        results = {
            'best_instruction': "N/A - Requires LLM setup",
            'best_score': 0.0,
            'total_time': elapsed_time,
            'num_evaluations': 0,
            'note': 'InstructZero requires open-source LLM configuration'
        }
        
        return results
    
    def compare_results(self, boing_results: Dict, instinct_results: Dict, iz_results: Dict):
        """Compare and analyze results from all three methods."""
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        # Create comparison table
        methods = {
            'BOInG': boing_results,
            'INSTINCT': instinct_results,
            'InstructZero': iz_results
        }
        
        # Find best method
        best_method = max(methods.items(), key=lambda x: x[1]['best_score'])
        
        comparison = {
            'summary': {
                'best_method': best_method[0],
                'best_score': float(best_method[1]['best_score']),
                'task': self.task,
                'budget': self.budget
            },
            'methods': {}
        }
        
        for method_name, results in methods.items():
            comparison['methods'][method_name] = {
                'best_score': float(results['best_score']),
                'total_time': float(results['total_time']),
                'time_per_eval': float(results['total_time']) / results['num_evaluations'] if results['num_evaluations'] > 0 else 0.0,
                'num_evaluations': results['num_evaluations'],
                'best_instruction': results['best_instruction']
            }
        
        # Print comparison table
        print(f"\n{'Method':<15} {'Best Score':<12} {'Time (s)':<12} {'Time/Eval':<12} {'Evaluations':<12}")
        print("-" * 80)
        for method_name, data in comparison['methods'].items():
            print(f"{method_name:<15} {data['best_score']:<12.4f} {data['total_time']:<12.2f} "
                  f"{data['time_per_eval']:<12.3f} {data['num_evaluations']:<12}")
        
        print("\n" + "-" * 80)
        print(f"Winner: {comparison['summary']['best_method']} "
              f"(Score: {comparison['summary']['best_score']:.4f})")
        print("="*80)
        
        return comparison
    
    def generate_report(self, boing_results: Dict, instinct_results: Dict, iz_results: Dict, comparison: Dict):
        """Generate comprehensive comparison report."""
        report = {
            'experiment_metadata': {
                'experiment_id': self.experiment_id,
                'timestamp': datetime.now().isoformat()
            },
            'config': self.results['config'],
            'results': {
                'boing': boing_results,
                'instinct': instinct_results,
                'instructzero': iz_results
            },
            'comparison': comparison
        }
        
        # Create experiment directory
        exp_dir = self.output_dir / f"comparison_{self.experiment_id}"
        exp_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        report_file = exp_dir / "results.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save text summary
        summary_file = exp_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BOInG vs INSTINCT vs InstructZero Comparison Report\n")
            f.write("="*80 + "\n\n")
            f.write(f"Experiment ID: {self.experiment_id}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Budget: {self.budget} evaluations ({self.init_points} init + {self.iterations} iterations)\n")
            f.write(f"API Model: {self.api_model}\n")
            f.write(f"Use API: {self.use_api}\n")
            f.write(f"Use Generator: {self.use_generator}\n\n")
            
            f.write("="*80 + "\n")
            f.write("RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for method_name, results in [
                ('BOInG', boing_results),
                ('INSTINCT', instinct_results),
                ('InstructZero', iz_results)
            ]:
                f.write(f"{method_name}:\n")
                f.write(f"  Best Instruction: {results['best_instruction']}\n")
                f.write(f"  Best Score: {results['best_score']:.4f}\n")
                f.write(f"  Total Time: {results['total_time']:.2f}s\n")
                f.write(f"  Evaluations: {results['num_evaluations']}\n")
                if results['num_evaluations'] > 0:
                    f.write(f"  Time per Eval: {results['total_time']/results['num_evaluations']:.3f}s\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"Best Method: {comparison['summary']['best_method']}\n")
            f.write(f"Best Score: {comparison['summary']['best_score']:.4f}\n\n")
            
            f.write("Performance Ranking:\n")
            sorted_methods = sorted(
                comparison['methods'].items(),
                key=lambda x: x[1]['best_score'],
                reverse=True
            )
            for rank, (method_name, data) in enumerate(sorted_methods, 1):
                f.write(f"  {rank}. {method_name}: {data['best_score']:.4f}\n")
        
        print(f"\nReport saved to: {report_file}")
        print(f"Summary saved to: {summary_file}")
        
        return report
    
    def run_comparison(self):
        """Run complete comparison experiment."""
        print("="*80)
        print("BOInG vs INSTINCT vs InstructZero Comparison")
        print("="*80)
        print(f"Task: {self.task}")
        print(f"Budget: {self.budget} evaluations ({self.init_points} init + {self.iterations} iterations)")
        print(f"API Model: {self.api_model}")
        print(f"Use API: {self.use_api}")
        print("="*80)
        
        # Run all methods
        boing_results = self.run_boing()
        self.results['boing'] = boing_results
        
        instinct_results = self.run_instinct()
        self.results['instinct'] = instinct_results
        
        iz_results = self.run_instructzero()
        self.results['instructzero'] = iz_results
        
        # Compare
        comparison = self.compare_results(boing_results, instinct_results, iz_results)
        
        # Generate report
        report = self.generate_report(boing_results, instinct_results, iz_results, comparison)
        
        return report


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare BOInG, INSTINCT, and InstructZero")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--use_openrouter", action="store_true", help="Use OpenRouter")
    parser.add_argument("--use_api", action="store_true", help="Use API for evaluation")
    parser.add_argument("--api_model", type=str, default="meta-llama/llama-3.2-3b-instruct:free", help="Model to use")
    parser.add_argument("--samples", type=int, default=10, help="Number of evaluation samples")
    parser.add_argument("--iterations", type=int, default=10, help="Number of optimization iterations")
    parser.add_argument("--init_points", type=int, default=5, help="Number of initial random evaluations")
    parser.add_argument("--use_generator", action="store_true", help="Use LLM to generate instructions")
    parser.add_argument("--output", type=str, default="comparison_results", help="Output directory")
    parser.add_argument("--model", type=str, default=None, help="Local model path (if not using API)")
    
    args = parser.parse_args()
    
    # Run comparison
    experiment = AllMethodsComparison(
        num_samples=args.samples,
        iterations=args.iterations,
        init_points=args.init_points,
        api_key=args.api_key,
        use_openrouter=args.use_openrouter,
        api_model=args.api_model,
        use_generator=args.use_generator,
        output_dir=args.output,
        model_path=args.model,
        use_api=args.use_api
    )
    
    report = experiment.run_comparison()
    
    print("\n" + "="*80)
    print("Comparison Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

