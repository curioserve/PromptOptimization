#!/usr/bin/env python3
"""
MATH500 Dataset Evaluator using GPT-OSS-20B model.
Supports multiple evaluation runs with statistical analysis.
"""

import json
import logging
import re
import time
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import requests
from transformers import pipeline
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MATH500Evaluator:
    def __init__(self, model_id: str = "./gpt-oss-20b", max_new_tokens: int = 1024, subject_filter: Optional[str] = None):
        """Initialize the MATH500 evaluator.
        
        Args:
            model_id: Path to the model or model identifier
            max_new_tokens: Maximum number of tokens to generate
            subject_filter: Optional subject filter (e.g., 'Algebra', 'Geometry')
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.subject_filter = subject_filter
        self.pipe = None
        self.all_runs_results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results") / f"math500_experiment_{self.experiment_id}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {self.results_dir}")
        
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
        """Load MATH500 dataset from HuggingFace."""
        logger.info("Loading MATH500 dataset from HuggingFace...")
        
        try:
            # Load the MATH-500 dataset from HuggingFace
            dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
            
            # Convert to list of dictionaries
            dataset_list = []
            for item in dataset:
                # Map HuggingFace format to our expected format
                dataset_item = {
                    "problem": item.get("problem", ""),
                    "solution": item.get("solution", ""),
                    "answer": item.get("answer", ""),
                    "subject": item.get("type", ""),  # HF uses 'type' instead of 'subject'
                    "level": item.get("level", 0),
                    "unique_id": item.get("unique_id", "")
                }
                dataset_list.append(dataset_item)
            
            # Apply subject filter if specified
            if self.subject_filter:
                dataset_list = [item for item in dataset_list if item.get('subject', '').lower() == self.subject_filter.lower()]
                logger.info(f"Filtered dataset to {len(dataset_list)} samples for subject: {self.subject_filter}")
            
            # Limit samples if specified
            if num_samples is not None:
                dataset_list = dataset_list[:num_samples]
                logger.info(f"Limited dataset to {len(dataset_list)} samples")
            
            logger.info(f"Loaded {len(dataset_list)} samples from MATH500 dataset")
            return dataset_list
            
        except Exception as e:
            logger.error(f"Failed to load MATH500 dataset from HuggingFace: {e}")
            # Fallback to local sample data if HuggingFace fails
            logger.info("Using fallback sample data...")
            return self._get_sample_data()[:num_samples] if num_samples else self._get_sample_data()
    
    def _get_sample_data(self) -> List[Dict[str, Any]]:
        """Fallback sample data based on the provided examples."""
        return [
            {
                "problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
                "solution": "We have that $r = \\sqrt{0^2 + 3^2} = 3.$ Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\\frac{\\pi}{2}$ with the positive $x$-axis. Therefore, the polar coordinates are $\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.$",
                "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
                "subject": "Precalculus",
                "level": 2,
                "unique_id": "test/precalculus/807.json"
            },
            {
                "problem": "If $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.",
                "solution": "$f(-2)+f(-1)+f(0)=\\frac{3(-2)-2}{-2-2}+\\frac{3(-1)-2}{-1-2}+\\frac{3(0)-2}{0-2}=\\frac{-8}{-4}+\\frac{-5}{-3}+\\frac{-2}{-2}=2+\\frac{5}{3}+1=\\boxed{\\frac{14}{3}}$",
                "answer": "\\frac{14}{3}",
                "subject": "Algebra",
                "level": 3,
                "unique_id": "test/algebra/2584.json"
            },
            {
                "problem": "How many positive whole-number divisors does 196 have?",
                "solution": "First prime factorize $196=2^2\\cdot7^2$. The prime factorization of any divisor of 196 cannot include any primes other than 2 and 7. We are free to choose either 0, 1, or 2 as the exponent of 2 in the prime factorization of a divisor of 196. Similarly, we may choose 0, 1, or 2 as the exponent of 7. In total, there are $3\\times 3=9$ possibilities for the prime factorization of a divisor of 196. Distinct prime factorizations correspond to distinct integers, so there are $\\boxed{9}$ divisors of 196.",
                "answer": "9",
                "subject": "Number Theory",
                "level": 3,
                "unique_id": "test/number_theory/572.json"
            }
        ]
    
    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single MATH500 sample."""
        problem = sample.get('problem', '')
        ground_truth = sample.get('answer', '')
        
        # Create messages for the model
        messages = [{"role": "user", "content": problem}]
        
        start_time = time.time()
        try:
            # Generate response using the pipeline
            outputs = self.pipe(
                messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            if isinstance(generated_text, list):
                # Get the assistant's response
                assistant_response = ""
                for msg in generated_text:
                    if msg.get("role") == "assistant":
                        assistant_response = msg.get("content", "")
                        break
                generated_text = assistant_response
            elif isinstance(generated_text, str):
                # If it's already a string, use it directly
                pass
            else:
                generated_text = str(generated_text)
            
            inference_time = time.time() - start_time
            
            # Extract the predicted answer
            predicted_answer = self.extract_answer(generated_text)
            
            # Check correctness
            is_correct = self.check_correctness(predicted_answer, ground_truth)
            
            result = {
                'problem': problem,
                'ground_truth': ground_truth,
                'generated_text': generated_text,
                'reasoning_chain': generated_text,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'inference_time': inference_time,
                'subject': sample.get('subject', ''),
                'level': sample.get('level', 0)
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            result = {
                'problem': problem,
                'ground_truth': ground_truth,
                'generated_text': "",
                'reasoning_chain': "",
                'predicted_answer': "",
                'is_correct': False,
                'inference_time': 0,
                'error': str(e),
                'subject': sample.get('subject', ''),
                'level': sample.get('level', 0)
            }
        
        return result
    
    def extract_answer(self, generated_text: str) -> str:
        """Extract the final answer from generated text."""
        # Look for boxed answers first (LaTeX format)
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        boxed_matches = re.findall(boxed_pattern, generated_text)
        if boxed_matches:
            return boxed_matches[-1].strip()
        
        # Look for answers in parentheses
        paren_pattern = r'\(([^)]*)\)(?=\s*$|\s*\.$)'
        paren_matches = re.findall(paren_pattern, generated_text)
        if paren_matches:
            return paren_matches[-1].strip()
        
        # Look for "Answer:" or "The answer is"
        answer_patterns = [
            r'(?:Answer|answer):\s*([^\n.]+)',
            r'(?:The answer is|the answer is)\s*([^\n.]+)',
            r'(?:Therefore|therefore),?\s*([^\n.]+)(?:\.|$)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, generated_text)
            if matches:
                return matches[-1].strip()
        
        # Look for mathematical expressions at the end
        math_pattern = r'([0-9]+(?:\.[0-9]+)?|\$[^$]+\$|\\frac\{[^}]+\}\{[^}]+\}|\\sqrt\{[^}]+\})'
        math_matches = re.findall(math_pattern, generated_text)
        if math_matches:
            return math_matches[-1].strip()
        
        # Return last line as fallback
        lines = generated_text.strip().split('\n')
        return lines[-1].strip() if lines else ""
    
    def check_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if the predicted answer matches the ground truth."""
        if not predicted or not ground_truth:
            return False
        
        # Normalize both answers
        pred_clean = re.sub(r'[^\w\d\.\-\+\(\)\[\]/\\]', '', predicted.lower())
        gt_clean = re.sub(r'[^\w\d\.\-\+\(\)\[\]/\\]', '', ground_truth.lower())
        
        # Direct string match
        if pred_clean == gt_clean:
            return True
        
        # Try to extract numeric values and compare
        try:
            pred_num = float(pred_clean)
            gt_num = float(gt_clean)
            return abs(pred_num - gt_num) < 1e-6
        except ValueError:
            pass
        
        # Check if prediction contains ground truth
        return gt_clean in pred_clean
    
    def evaluate_single_run(self, num_samples: Optional[int] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
        
        # Calculate summary statistics
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
    
    def save_results(self, output_path: Optional[str] = None):
        """Save comprehensive evaluation results with detailed per-run tracking."""
        if not self.all_runs_results:
            logger.warning("No results to save. Run evaluation first.")
            return
        
        # Use experiment-specific directory if no path provided
        if output_path is None:
            output_path = self.results_dir / "complete_results.json"
        
        # Calculate overall statistics
        all_accuracies = [run['summary']['accuracy'] for run in self.all_runs_results]
        all_inference_times = [run['summary']['avg_inference_time'] for run in self.all_runs_results]
        
        # Create comprehensive output data
        output_data = {
            'experiment_metadata': {
                'experiment_id': self.experiment_id,
                'timestamp': datetime.now().isoformat(),
                'model_id': self.model_id,
                'max_new_tokens': self.max_new_tokens,
                'subject_filter': self.subject_filter,
                'num_runs': len(self.all_runs_results),
                'samples_per_run': len(self.all_runs_results[0]['results']) if self.all_runs_results else 0,
                'total_samples_evaluated': sum(len(run['results']) for run in self.all_runs_results)
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
        
        # Save main results file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save individual run files
        self._save_individual_run_files()
        
        # Create cross-run comparison table
        self._create_cross_run_comparison()
        
        # Create summary statistics file
        self._create_summary_statistics()
        
        # Create subject-wise analysis if applicable
        self._create_subject_analysis()
        
        logger.info(f"Comprehensive results saved to: {self.results_dir}")
        logger.info(f"Main results file: {output_path}")
    
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
            print(f"Problem: {result['problem'][:100]}...")
            print(f"Subject: {result['subject']} (Level {result['level']})")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Predicted: {result['predicted_answer']}")
            print(f"Correct: ✅" if result['is_correct'] else "❌")
            print(f"Time: {result['inference_time']:.2f}s")
            
            if 'error' in result:
                print(f"Error: {result['error']}")


    def _save_individual_run_files(self):
        """Save detailed results for each individual run."""
        runs_dir = self.results_dir / "individual_runs"
        runs_dir.mkdir(exist_ok=True)
        
        for run_data in self.all_runs_results:
            run_num = run_data['run_number']
            run_file = runs_dir / f"run_{run_num:02d}_detailed.json"
            
            with open(run_file, 'w') as f:
                json.dump(run_data, f, indent=2)
            
            # Also save as CSV for easy analysis
            csv_file = runs_dir / f"run_{run_num:02d}_results.csv"
            df = pd.DataFrame(run_data['results'])
            df.to_csv(csv_file, index=False)
    
    def _create_cross_run_comparison(self):
        """Create a table showing which questions were answered correctly in each run."""
        if not self.all_runs_results or len(self.all_runs_results) < 2:
            return
        
        # Get all questions (assuming same questions across runs)
        first_run_results = self.all_runs_results[0]['results']
        num_questions = len(first_run_results)
        
        # Create comparison data
        comparison_data = []
        for i in range(num_questions):
            row = {
                'question_id': i + 1,
                'problem': first_run_results[i]['problem'][:100] + "...",
                'subject': first_run_results[i]['subject'],
                'level': first_run_results[i]['level'],
                'ground_truth': first_run_results[i]['ground_truth']
            }
            
            # Add correctness for each run
            for run_data in self.all_runs_results:
                run_num = run_data['run_number']
                is_correct = run_data['results'][i]['is_correct']
                predicted = run_data['results'][i]['predicted_answer']
                row[f'run_{run_num}_correct'] = '✓' if is_correct else '✗'
                row[f'run_{run_num}_answer'] = predicted
            
            # Calculate consistency metrics
            correct_runs = sum(1 for run_data in self.all_runs_results 
                             if run_data['results'][i]['is_correct'])
            row['correct_count'] = correct_runs
            row['consistency'] = f"{correct_runs}/{len(self.all_runs_results)}"
            row['is_consistent'] = correct_runs == 0 or correct_runs == len(self.all_runs_results)
            
            comparison_data.append(row)
        
        # Save as CSV
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = self.results_dir / "cross_run_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        # Save inconsistent questions separately
        inconsistent_df = comparison_df[~comparison_df['is_consistent']]
        if not inconsistent_df.empty:
            inconsistent_file = self.results_dir / "inconsistent_questions.csv"
            inconsistent_df.to_csv(inconsistent_file, index=False)
            logger.info(f"Found {len(inconsistent_df)} inconsistent questions (saved to inconsistent_questions.csv)")
        
        logger.info(f"Cross-run comparison saved to: {comparison_file}")
    
    def _create_summary_statistics(self):
        """Create a summary statistics file with key metrics."""
        if not self.all_runs_results:
            return
        
        summary_stats = {
            'experiment_overview': {
                'experiment_id': self.experiment_id,
                'model': self.model_id,
                'subject_filter': self.subject_filter,
                'num_runs': len(self.all_runs_results),
                'samples_per_run': len(self.all_runs_results[0]['results']) if self.all_runs_results else 0
            },
            'accuracy_analysis': {
                'per_run_accuracy': [run['summary']['accuracy'] for run in self.all_runs_results],
                'mean_accuracy': sum(run['summary']['accuracy'] for run in self.all_runs_results) / len(self.all_runs_results),
                'accuracy_std': self._calculate_std([run['summary']['accuracy'] for run in self.all_runs_results]),
                'min_accuracy': min(run['summary']['accuracy'] for run in self.all_runs_results),
                'max_accuracy': max(run['summary']['accuracy'] for run in self.all_runs_results)
            },
            'timing_analysis': {
                'per_run_avg_time': [run['summary']['avg_inference_time'] for run in self.all_runs_results],
                'overall_avg_time': sum(run['summary']['avg_inference_time'] for run in self.all_runs_results) / len(self.all_runs_results),
                'time_std': self._calculate_std([run['summary']['avg_inference_time'] for run in self.all_runs_results])
            }
        }
        
        # Add consistency analysis if multiple runs
        if len(self.all_runs_results) > 1:
            # Calculate question-level consistency
            num_questions = len(self.all_runs_results[0]['results'])
            consistent_questions = 0
            
            for i in range(num_questions):
                correct_runs = sum(1 for run_data in self.all_runs_results 
                                 if run_data['results'][i]['is_correct'])
                if correct_runs == 0 or correct_runs == len(self.all_runs_results):
                    consistent_questions += 1
            
            summary_stats['consistency_analysis'] = {
                'consistent_questions': consistent_questions,
                'inconsistent_questions': num_questions - consistent_questions,
                'consistency_rate': consistent_questions / num_questions if num_questions > 0 else 0
            }
        
        # Save summary
        summary_file = self.results_dir / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info(f"Summary statistics saved to: {summary_file}")
    
    def _create_subject_analysis(self):
        """Create subject-wise performance analysis."""
        if not self.all_runs_results:
            return
        
        # Collect all results across runs
        all_results = []
        for run_data in self.all_runs_results:
            for result in run_data['results']:
                result_copy = result.copy()
                result_copy['run_number'] = run_data['run_number']
                all_results.append(result_copy)
        
        df = pd.DataFrame(all_results)
        
        if 'subject' in df.columns:
            # Subject-wise accuracy
            subject_stats = df.groupby('subject').agg({
                'is_correct': ['count', 'sum', 'mean'],
                'inference_time': 'mean',
                'level': 'mean'
            }).round(3)
            
            subject_stats.columns = ['total_questions', 'correct_answers', 'accuracy', 'avg_inference_time', 'avg_level']
            subject_file = self.results_dir / "subject_analysis.csv"
            subject_stats.to_csv(subject_file)
            
            # Level-wise accuracy
            if 'level' in df.columns:
                level_stats = df.groupby('level').agg({
                    'is_correct': ['count', 'sum', 'mean'],
                    'inference_time': 'mean'
                }).round(3)
                
                level_stats.columns = ['total_questions', 'correct_answers', 'accuracy', 'avg_inference_time']
                level_file = self.results_dir / "level_analysis.csv"
                level_stats.to_csv(level_file)
            
            logger.info(f"Subject and level analysis saved to CSV files")


def main():
    """Main evaluation function."""
    # Configuration
    MODEL_ID = "./gpt-oss-20b"
    NUM_SAMPLES = 20  # Set to None to evaluate all samples
    NUM_RUNS = 3     # Number of evaluation runs
    MAX_NEW_TOKENS = 1024
    
    # Initialize evaluator
    evaluator = MATH500Evaluator(model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS)
    logger.info(f"Starting MATH500 evaluation experiment: {evaluator.experiment_id}")
    
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
