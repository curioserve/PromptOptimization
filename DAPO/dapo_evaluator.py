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
    def __init__(self, model_id: str = "openai/gpt-oss-20b", max_new_tokens: int = 512, dataset_config: str = "all"):
        """Initialize the DAPO evaluator with model pipeline."""
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.dataset_config = dataset_config
        self.pipe = None
        self.results = []
        
    def setup_pipeline(self):
        """Initialize the model pipeline."""
        logger.info(f"Loading model: {self.model_id}")
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
        # Look for "Answer: " pattern
        answer_match = re.search(r"Answer:\s*([^\n]+)", text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Look for final numerical answer or expression
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('Therefore') and not line.startswith('Thus'):
                # Try to extract numerical answer
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
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )
            inference_time = time.time() - start_time
            
            # Extract generated text
            if isinstance(outputs[0]["generated_text"], list):
                generated_text = outputs[0]["generated_text"][-1]["content"]
            else:
                generated_text = outputs[0]["generated_text"]
            
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
    
    def evaluate_dataset(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate the entire dataset."""
        if self.pipe is None:
            self.setup_pipeline()
        
        # Load dataset
        dataset = self.load_dataset(num_samples)
        
        # Process samples
        self.results = []
        correct_count = 0
        total_inference_time = 0
        
        logger.info(f"Starting evaluation of {len(dataset)} samples...")
        
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
            result = self.evaluate_single(sample)
            self.results.append(result)
            
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
        
        return summary
    
    def save_results(self, output_path: str = "dapo_evaluation_results.json"):
        """Save evaluation results to JSON file."""
        output_data = {
            'summary': {
                'total_samples': len(self.results),
                'correct_predictions': sum(1 for r in self.results if r['is_correct']),
                'accuracy': sum(1 for r in self.results if r['is_correct']) / len(self.results) if self.results else 0,
                'model_id': self.model_id,
                'max_new_tokens': self.max_new_tokens
            },
            'detailed_results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_sample_results(self, num_samples: int = 5):
        """Print a few sample results for inspection."""
        if not self.results:
            logger.warning("No results available")
            return
        
        print(f"\n{'='*80}")
        print(f"SAMPLE RESULTS (showing first {min(num_samples, len(self.results))} samples)")
        print(f"{'='*80}")
        
        for i, result in enumerate(self.results[:num_samples]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {result['prompt'][:100]}...")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Predicted: {result['predicted_answer']}")
            print(f"Correct: {result['is_correct']}")
            print(f"Time: {result['inference_time']:.2f}s")
            if 'error' in result:
                print(f"Error: {result['error']}")


def main():
    """Main evaluation function."""
    # Configuration
    MODEL_ID = "openai/gpt-oss-20b"
    NUM_SAMPLES = 50  # Set to None to evaluate all samples
    MAX_NEW_TOKENS = 512
    
    # Initialize evaluator
    evaluator = DAPOEvaluator(model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS)
    
    try:
        # Run evaluation
        summary = evaluator.evaluate_dataset(num_samples=NUM_SAMPLES)
        
        # Save results
        evaluator.save_results()
        
        # Print sample results
        evaluator.print_sample_results()
        
        # Print final summary
        print(f"\n{'='*80}")
        print("FINAL EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {summary['model_id']}")
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Correct Predictions: {summary['correct_predictions']}")
        print(f"Accuracy: {summary['accuracy']:.3f}")
        print(f"Average Inference Time: {summary['avg_inference_time']:.2f}s")
        print(f"Total Inference Time: {summary['total_inference_time']:.1f}s")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
