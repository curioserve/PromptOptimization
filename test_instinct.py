#!/usr/bin/env python3
"""
Test INSTINCT (INSTruction optimization usIng Neural bandits Coupled with Transformers) with MATH500 dataset.

This script:
1. Sets up GPT-2 embeddings for the INSTINCT optimizer
2. Creates a scoring function that evaluates instructions on MATH500 dataset
3. Optionally uses an LLM API to generate instructions from seed prompts
4. Runs INSTINCT optimization to find the best instruction
"""

import os
import sys
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from INSTINCT import INSTINCT
from MATH500.math500_evaluator import MATH500Evaluator

# Optional: Import OpenAI for instruction generation
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Instruction generation will use seed prompts directly.")


def setup_gpt2_embeddings():
    """Load GPT-2 tokenizer and model to get embeddings."""
    print("Loading GPT-2 tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    
    # Get embedding matrix: shape (vocab_size, embed_dim)
    embedding_matrix = model.wte.weight.detach().numpy()  # shape (50257, 768)
    vocab_tokens = list(tokenizer.get_vocab().keys())  # list of 50257 tokens
    
    print(f"Loaded embeddings: {embedding_matrix.shape[0]} tokens, {embedding_matrix.shape[1]} dimensions")
    return embedding_matrix, vocab_tokens, tokenizer


def create_scoring_function_api(api_key, model_name="meta-llama/llama-3.2-3b-instruct:free", num_samples=10, max_tokens=1024, use_openrouter=False):
    """
    Create a scoring function that evaluates an instruction on MATH500 dataset using API.
    
    Args:
        api_key: API key (OpenAI or OpenRouter)
        model_name: Model name to use
        num_samples: Number of MATH500 samples to evaluate on
        max_tokens: Maximum tokens to generate
        use_openrouter: If True, use OpenRouter API instead of OpenAI
    
    Returns:
        A function that takes an instruction string and returns a score (accuracy)
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package is required for API-based evaluation. Install with: pip install openai")
    
    # Set up API key
    if api_key:
        api_key_value = api_key
    elif os.getenv("OPENAI_API_KEY") and not use_openrouter:
        api_key_value = os.getenv("OPENAI_API_KEY")
    elif os.getenv("OPENROUTER_API_KEY") and use_openrouter:
        api_key_value = os.getenv("OPENROUTER_API_KEY")
    else:
        if use_openrouter:
            raise ValueError("OpenRouter API key is required. Provide via --api_key or OPENROUTER_API_KEY environment variable.")
        else:
            raise ValueError("OpenAI API key is required. Provide via --api_key or OPENAI_API_KEY environment variable.")
    
    # Set base URL for OpenRouter or OpenAI
    if use_openrouter:
        base_url = "https://openrouter.ai/api/v1"
        print(f"Using OpenRouter API with model: {model_name}")
    else:
        base_url = "https://api.openai.com/v1"
        print(f"Using OpenAI API with model: {model_name}")
    
    # Create OpenAI client
    client = openai.OpenAI(api_key=api_key_value, base_url=base_url)
    
    # Load dataset using evaluator
    from MATH500.math500_evaluator import MATH500Evaluator
    evaluator = MATH500Evaluator(model_id="microsoft/DialoGPT-small", max_new_tokens=max_tokens)
    evaluator._api_mode = True
    evaluator.pipe = None
    
    # Cache the dataset
    if not hasattr(create_scoring_function_api, '_dataset'):
        print(f"Loading MATH500 dataset (first {num_samples} samples)...")
        create_scoring_function_api._dataset = evaluator.load_dataset(num_samples=num_samples)
        print(f"Loaded {len(create_scoring_function_api._dataset)} samples")
    
    dataset = create_scoring_function_api._dataset
    
    def score_instruction(instruction: str) -> float:
        """Evaluate an instruction on MATH500 dataset using OpenAI API."""
        correct_count = 0
        total = len(dataset)
        
        for sample in dataset:
            problem = sample.get('problem', '')
            ground_truth = sample.get('answer', '')
            
            prompt = f"{instruction}\n\nProblem: {problem}\nSolution:"
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens
                )
                
                generated_text = response.choices[0].message.content
                predicted_answer = evaluator.extract_answer(generated_text)
                is_correct = evaluator.check_correctness(predicted_answer, ground_truth)
                
                if is_correct:
                    correct_count += 1
                    
            except Exception as e:
                print(f"Error evaluating sample: {e}")
        
        accuracy = correct_count / total if total > 0 else 0.0
        print(f"  Instruction: '{instruction[:50]}...' -> Accuracy: {accuracy:.3f} ({correct_count}/{total})")
        return accuracy
    
    return score_instruction


def create_scoring_function(evaluator, num_samples=10):
    """
    Create a scoring function that evaluates an instruction on MATH500 dataset (local model).
    """
    if not hasattr(create_scoring_function, '_dataset'):
        print(f"Loading MATH500 dataset (first {num_samples} samples)...")
        create_scoring_function._dataset = evaluator.load_dataset(num_samples=num_samples)
        print(f"Loaded {len(create_scoring_function._dataset)} samples")
    
    dataset = create_scoring_function._dataset
    
    def score_instruction(instruction: str) -> float:
        """Evaluate an instruction on MATH500 dataset."""
        if hasattr(evaluator, '_api_mode') and evaluator._api_mode:
            pass
        elif not evaluator.pipe:
            evaluator.setup_pipeline()
        
        correct_count = 0
        total = len(dataset)
        
        for sample in dataset:
            problem = sample.get('problem', '')
            ground_truth = sample.get('answer', '')
            prompt = f"{instruction}\n\nProblem: {problem}\nSolution:"
            
            try:
                messages = [{"role": "user", "content": prompt}]
                outputs = evaluator.pipe(
                    messages,
                    max_new_tokens=evaluator.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=evaluator.pipe.tokenizer.eos_token_id
                )
                
                generated_text = outputs[0]["generated_text"]
                if isinstance(generated_text, list):
                    assistant_response = ""
                    for msg in generated_text:
                        if msg.get("role") == "assistant":
                            assistant_response = msg.get("content", "")
                            break
                    generated_text = assistant_response
                
                predicted_answer = evaluator.extract_answer(generated_text)
                is_correct = evaluator.check_correctness(predicted_answer, ground_truth)
                
                if is_correct:
                    correct_count += 1
                    
            except Exception as e:
                print(f"Error evaluating sample: {e}")
        
        accuracy = correct_count / total if total > 0 else 0.0
        print(f"  Instruction: '{instruction[:50]}...' -> Accuracy: {accuracy:.3f} ({correct_count}/{total})")
        return accuracy
    
    return score_instruction


def create_generator_function(api_key=None, model="gpt-3.5-turbo", use_openrouter=False):
    """Create a function that generates instructions from seed prompts using an LLM."""
    if not OPENAI_AVAILABLE:
        print("OpenAI not available. Using identity function (seed prompt = instruction).")
        return lambda seed_prompt: seed_prompt
    
    if api_key:
        api_key_value = api_key
    elif os.getenv("OPENAI_API_KEY") and not use_openrouter:
        api_key_value = os.getenv("OPENAI_API_KEY")
    elif os.getenv("OPENROUTER_API_KEY") and use_openrouter:
        api_key_value = os.getenv("OPENROUTER_API_KEY")
    else:
        print("No API key provided. Using identity function (seed prompt = instruction).")
        return lambda seed_prompt: seed_prompt
    
    if use_openrouter:
        base_url = "https://openrouter.ai/api/v1"
    else:
        base_url = "https://api.openai.com/v1"
    
    client = openai.OpenAI(api_key=api_key_value, base_url=base_url)
    
    def generate_instruction(seed_prompt: str) -> str:
        """Generate an instruction from a seed prompt using LLM."""
        generation_prompt = f"""Given these seed words: {seed_prompt}

Generate a clear, concise instruction for solving math problems. The instruction should guide a language model to solve mathematical problems step by step.

Instruction:"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": generation_prompt}],
                temperature=0.7,
                max_tokens=100
            )
            instruction = response.choices[0].message.content.strip()
            return instruction
        except Exception as e:
            print(f"Error generating instruction: {e}. Using seed prompt as instruction.")
            return seed_prompt
    
    return generate_instruction


def main():
    """Main function to run INSTINCT optimization with MATH500."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test INSTINCT with MATH500 dataset")
    parser.add_argument("--model", default="./gpt-oss-20b", help="Model ID for MATH500 evaluation")
    parser.add_argument("--samples", type=int, default=10, help="Number of MATH500 samples to evaluate on")
    parser.add_argument("--iterations", type=int, default=10, help="Number of optimization iterations")
    parser.add_argument("--init_points", type=int, default=5, help="Number of initial random evaluations")
    parser.add_argument("--n_tokens", type=int, default=5, help="Number of tokens in seed prompt")
    parser.add_argument("--latent_dim", type=int, default=50, help="Latent dimension for optimization")
    parser.add_argument("--api_key", type=str, default=None, help="API key for model inference")
    parser.add_argument("--use_api", action="store_true", help="Use API for model inference")
    parser.add_argument("--use_openrouter", action="store_true", help="Use OpenRouter API")
    parser.add_argument("--api_model", type=str, default="meta-llama/llama-3.2-3b-instruct:free", help="Model to use")
    parser.add_argument("--use_generator", action="store_true", help="Use LLM to generate instructions")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for MATH500 evaluation")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="Regularization parameter")
    parser.add_argument("--nu", type=float, default=0.1, help="Exploration parameter")
    parser.add_argument("--local_training_iter", type=int, default=30, help="Neural network training iterations")
    parser.add_argument("--n_domain", type=int, default=1000, help="Number of candidate points")
    parser.add_argument("--n_eval", type=int, default=100, help="Number of points to evaluate per iteration")
    
    args = parser.parse_args()
    
    print("="*80)
    print("INSTINCT + MATH500 Test")
    print("="*80)
    if args.use_api:
        api_provider = "OpenRouter" if args.use_openrouter else "OpenAI"
        print(f"Model: {args.api_model} (via {api_provider} API)")
    else:
        print(f"Model: {args.model} (local)")
    print(f"MATH500 samples: {args.samples}")
    print(f"Optimization iterations: {args.iterations}")
    print(f"Initial points: {args.init_points}")
    print(f"Seed prompt tokens: {args.n_tokens}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Lambda (regularization): {args.lambda_reg}")
    print(f"Nu (exploration): {args.nu}")
    print("="*80)
    
    # 1. Set up GPT-2 embeddings
    embedding_matrix, vocab_tokens, tokenizer = setup_gpt2_embeddings()
    
    # 2. Create scoring function
    print("\nCreating scoring function...")
    if args.use_api:
        scoring_func = create_scoring_function_api(
            api_key=args.api_key,
            model_name=args.api_model,
            num_samples=args.samples,
            max_tokens=args.max_tokens,
            use_openrouter=args.use_openrouter
        )
    else:
        model_path = args.model
        if model_path.startswith('./') or (os.path.exists(model_path) and not os.path.isabs(model_path)):
            model_path = os.path.abspath(model_path)
        evaluator = MATH500Evaluator(
            model_id=model_path,
            max_new_tokens=args.max_tokens
        )
        scoring_func = create_scoring_function(evaluator, num_samples=args.samples)
    
    # 3. Create generator function (optional)
    generator_func = None
    if args.use_generator:
        print("\nSetting up instruction generator...")
        generator_func = create_generator_function(api_key=args.api_key, use_openrouter=args.use_openrouter)
    else:
        print("\nSkipping instruction generator (using seed prompts directly)")
    
    # 4. Initialize INSTINCT optimizer
    print("\nInitializing INSTINCT optimizer...")
    instinct = INSTINCT(
        scoring_func=scoring_func,
        embedding_matrix=embedding_matrix,
        token_list=vocab_tokens,
        generator_func=generator_func,
        n_tokens=args.n_tokens,
        latent_dim=args.latent_dim,
        lambda_reg=args.lambda_reg,
        nu=args.nu,
        local_training_iter=args.local_training_iter,
        n_domain=args.n_domain,
        n_eval=args.n_eval
    )
    
    # 5. Run optimization
    print("\n" + "="*80)
    print("Starting INSTINCT optimization...")
    print("="*80)
    best_instruction, best_score = instinct.optimize(
        iterations=args.iterations,
        init_points=args.init_points,
        verbose=True
    )
    
    # 6. Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best instruction: {best_instruction}")
    print(f"Best score (accuracy): {best_score:.4f}")
    print("="*80)
    
    return best_instruction, best_score


if __name__ == "__main__":
    main()

