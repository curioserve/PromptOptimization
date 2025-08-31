"""
DAPO Math Dataset Evaluator using OpenAI GPT-OSS-20B
Loads dataset, runs inference, and evaluates results against ground truth.
"""

import json
import re
import torch
from transformers import pipeline
from datasets import load_dataset
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DAPOEvaluator:
    def __init__(self, model_id: str = "./gpt-oss-20b", max_new_tokens: int = 1024, dataset_config: str = "all"):
        """Initialize the DAPO evaluator with model pipeline."""
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.dataset_config = dataset_config
        self.pipe = None
        self.all_runs_results = []  # Store results from all runs
        
    def setup_pipeline(self):
        """Initialize the model pipeline with harmony format support."""
        logger.info(f"Loading model: {self.model_id}")
        logger.info("Using transformers pipeline - harmony format applied automatically")
        try:
            self.pipe = pipeline(
                "text-generation",
                model=self.model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Model pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_dataset(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load DAPO Math dataset."""
        logger.info(f"Loading DAPO Math dataset with config '{self.dataset_config}'...")
        try:
            dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", self.dataset_config, split="train")
            
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
            logger.info(f"Loaded {len(dataset)} samples")
            return list(dataset)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from model output."""
        # Multiple answer extraction patterns for better accuracy
        patterns = [
            r"Answer:\s*([^\n]+)",
            r"(?:Therefore|Thus|Hence),?\s*(?:the answer is|answer:)?\s*([^\n]+)",
            r"\\boxed\{([^}]+)\}",  # LaTeX boxed format
            r"=\s*([0-9]+(?:\.[0-9]+)?)\s*$",  # Ends with = number
            r"\b(\d+)\s*$",  # Ends with a number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, try to extract from last meaningful line
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) < 50:  # Likely an answer line
                num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', line)
                if num_match:
                    return num_match.group(1)
        
        return None
    
    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample."""
        prompt = sample.get('prompt', '')
        ground_truth = sample.get('solution', '')
        source_prompt = sample.get('source_prompt', [])
        
        # Use source_prompt if available, otherwise construct from prompt
        if source_prompt and isinstance(source_prompt, list) and len(source_prompt) > 0:
            messages = source_prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        
        try:
            # Generate response
            start_time = time.time()
            outputs = self.pipe(
                messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Use greedy decoding for consistent reasoning
                temperature=1.0,  # Standard temperature for reasoning models
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )
            inference_time = time.time() - start_time
            
            # Extract generated text
            if isinstance(outputs[0]["generated_text"], list):
                generated_text = outputs[0]["generated_text"][-1]["content"]
            else:
                generated_text = outputs[0]["generated_text"]
                # Remove input from output if present
                for msg in messages:
                    if msg["content"] in generated_text:
                        generated_text = generated_text.replace(msg["content"], "").strip()
            
            # Extract answer
            predicted_answer = self.extract_answer(generated_text)
            
            # Evaluate correctness
            is_correct = self.check_correctness(predicted_answer, ground_truth)
            
            result = {
                'prompt': prompt,
                'ground_truth': ground_truth,
                'generated_text': generated_text,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'inference_time': inference_time,
                'data_source': sample.get('data_source', ''),
                'ability': sample.get('ability', ''),
                'index': sample.get('extra_info', {}).get('index', '')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            return {
                'prompt': prompt,
                'ground_truth': ground_truth,
                'generated_text': '',
                'predicted_answer': None,
                'is_correct': False,
                'inference_time': 0,
                'error': str(e),
                'data_source': sample.get('data_source', ''),
                'ability': sample.get('ability', ''),
                'index': sample.get('extra_info', {}).get('index', '')
            }
    
    def check_correctness(self, predicted: Optional[str], ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        if predicted is None:
            return False
        
        # Clean both answers
        pred_clean = str(predicted).strip().lower()
        gt_clean = str(ground_truth).strip().lower()
        
        # Direct match
        if pred_clean == gt_clean:
            return True
        
        # Try numerical comparison
        try:
            pred_num = float(pred_clean)
            gt_num = float(gt_clean)
            return abs(pred_num - gt_num) < 1e-6
        except ValueError:
            pass
        
        # Check if prediction contains ground truth
        return gt_clean in pred_clean
    
    def evaluate_single_run(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate the dataset in a single run."""
        if self.pipe is None:
            self.setup_pipeline()
        
        # Load dataset
        dataset = self.load_dataset(num_samples)
        
        # Process samples
        results = []
        correct_count = 0
        total_inference_time = 0
        
        logger.info(f"Starting single run evaluation of {len(dataset)} samples...")
        
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
            result = self.evaluate_single(sample)
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            total_inference_time += result.get('inference_time', 0)
            
            # Log progress every 10 samples
            if (i + 1) % 10 == 0:
                current_accuracy = correct_count / (i + 1)
                logger.info(f"Progress: {i+1}/{len(dataset)}, Accuracy: {current_accuracy:.3f}")
        
        # Calculate final metrics
        accuracy = correct_count / len(dataset) if dataset else 0
        avg_inference_time = total_inference_time / len(dataset) if dataset else 0
        
        summary = {
            'total_samples': len(dataset),
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'total_inference_time': total_inference_time,
            'model_id': self.model_id
        }
        
        logger.info(f"Evaluation complete!")
        logger.info(f"Accuracy: {accuracy:.3f} ({correct_count}/{len(dataset)})")
        logger.info(f"Average inference time: {avg_inference_time:.2f}s")
        
        return summary, results
    
    def evaluate_multiple_runs(self, num_samples: Optional[int] = None, num_runs: int = 1) -> Dict[str, Any]:
        """Evaluate the dataset multiple times and calculate statistics."""
        logger.info(f"Starting {num_runs} evaluation runs...")
        
        all_runs_data = []
        all_accuracies = []
        all_inference_times = []
        all_correct_counts = []
        
        for run_idx in range(num_runs):
            logger.info(f"\n--- Run {run_idx + 1}/{num_runs} ---")
            summary, results = self.evaluate_single_run(num_samples)
            
            run_data = {
                'run_number': run_idx + 1,
                'summary': summary,
                'results': results
            }
            
            all_runs_data.append(run_data)
            all_accuracies.append(summary['accuracy'])
            all_inference_times.append(summary['avg_inference_time'])
            all_correct_counts.append(summary['correct_predictions'])
        
        # Calculate overall statistics
        overall_stats = {
            'num_runs': num_runs,
            'samples_per_run': num_samples or len(all_runs_data[0]['results']),
            'total_samples_evaluated': sum(len(run['results']) for run in all_runs_data),
            'accuracy_stats': {
                'mean': sum(all_accuracies) / len(all_accuracies),
                'min': min(all_accuracies),
                'max': max(all_accuracies),
                'std': self._calculate_std(all_accuracies),
                'all_runs': all_accuracies
            },
            'inference_time_stats': {
                'mean': sum(all_inference_times) / len(all_inference_times),
                'min': min(all_inference_times),
                'max': max(all_inference_times),
                'std': self._calculate_std(all_inference_times),
                'all_runs': all_inference_times
            },
            'correct_predictions_stats': {
                'mean': sum(all_correct_counts) / len(all_correct_counts),
                'min': min(all_correct_counts),
                'max': max(all_correct_counts),
                'std': self._calculate_std(all_correct_counts),
                'all_runs': all_correct_counts
            }
        }
        
        # Store all results
        self.all_runs_results = all_runs_data
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MULTIPLE RUNS SUMMARY ({num_runs} runs)")
        logger.info(f"{'='*60}")
        logger.info(f"Average Accuracy: {overall_stats['accuracy_stats']['mean']:.3f} ± {overall_stats['accuracy_stats']['std']:.3f}")
        logger.info(f"Accuracy Range: {overall_stats['accuracy_stats']['min']:.3f} - {overall_stats['accuracy_stats']['max']:.3f}")
        logger.info(f"Average Inference Time: {overall_stats['inference_time_stats']['mean']:.2f}s ± {overall_stats['inference_time_stats']['std']:.2f}s")
        
        return overall_stats
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def save_results(self, output_path: str = "dapo_evaluation_results.json"):
        """Save evaluation results to JSON file."""
        if not self.all_runs_results:
            logger.warning("No results to save. Run evaluation first.")
            return
        
        # Calculate overall statistics
        all_accuracies = [run['summary']['accuracy'] for run in self.all_runs_results]
        all_inference_times = [run['summary']['avg_inference_time'] for run in self.all_runs_results]
        
        output_data = {
            'evaluation_metadata': {
                'model_id': self.model_id,
                'max_new_tokens': self.max_new_tokens,
                'dataset_config': self.dataset_config,
                'num_runs': len(self.all_runs_results),
                'samples_per_run': len(self.all_runs_results[0]['results']) if self.all_runs_results else 0
            },
            'overall_statistics': {
                'accuracy': {
                    'mean': sum(all_accuracies) / len(all_accuracies),
                    'min': min(all_accuracies),
                    'max': max(all_accuracies),
                    'std': self._calculate_std(all_accuracies),
                    'all_runs': all_accuracies
                },
                'inference_time': {
                    'mean': sum(all_inference_times) / len(all_inference_times),
                    'min': min(all_inference_times),
                    'max': max(all_inference_times),
                    'std': self._calculate_std(all_inference_times),
                    'all_runs': all_inference_times
                }
            },
            'individual_runs': self.all_runs_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_sample_results(self, num_samples: int = 5, run_number: int = 1):
        """Print a few sample results for inspection."""
        if not self.all_runs_results:
            logger.warning("No results available")
            return
        
        if run_number > len(self.all_runs_results):
            logger.warning(f"Run {run_number} not found. Available runs: 1-{len(self.all_runs_results)}")
            return
        
        results = self.all_runs_results[run_number - 1]['results']
        
        print(f"\n{'='*80}")
        print(f"SAMPLE RESULTS - Run {run_number} (showing first {min(num_samples, len(results))} samples)")
        print(f"{'='*80}")
        
        for i, result in enumerate(results[:num_samples]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {result['prompt'][:100]}...")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Predicted: {result['predicted_answer']}")
            print(f"Correct: ✅" if result['is_correct'] else "❌")
            print(f"Time: {result['inference_time']:.2f}s")
            
            if 'error' in result:
                print(f"Error: {result['error']}")


def main():
    """Main evaluation function."""
    # Configuration
    MODEL_ID = "./gpt-oss-20b"
    NUM_SAMPLES = 20  # Set to None to evaluate all samples
    NUM_RUNS = 3     # Number of evaluation runs
    MAX_NEW_TOKENS = 1024
    
    # Initialize evaluator
    evaluator = DAPOEvaluator(model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS)
    
    try:
        # Run multiple evaluations
        overall_stats = evaluator.evaluate_multiple_runs(num_samples=NUM_SAMPLES, num_runs=NUM_RUNS)
        
        # Save results
        evaluator.save_results()
        
        # Print sample results from first run
        evaluator.print_sample_results(num_samples=3, run_number=1)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("FINAL MULTIPLE RUNS SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {MODEL_ID}")
        print(f"Number of Runs: {NUM_RUNS}")
        print(f"Samples per Run: {NUM_SAMPLES}")
        print(f"Average Accuracy: {overall_stats['accuracy_stats']['mean']:.3f} ± {overall_stats['accuracy_stats']['std']:.3f}")
        print(f"Accuracy Range: {overall_stats['accuracy_stats']['min']:.3f} - {overall_stats['accuracy_stats']['max']:.3f}")
        print(f"Average Inference Time: {overall_stats['inference_time_stats']['mean']:.2f}s ± {overall_stats['inference_time_stats']['std']:.2f}s")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
