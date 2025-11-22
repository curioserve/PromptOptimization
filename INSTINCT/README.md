# INSTINCT: INSTruction optimization usIng Neural bandits Coupled with Transformers

This is an implementation of INSTINCT (ICML 2024) integrated into the prompt optimization framework.

## Overview

INSTINCT uses **neural bandits** (NeuralTSDiag) instead of Gaussian Processes for instruction optimization. The key differences from BOInG:

- **Surrogate Model**: Neural network instead of Gaussian Process
- **Optimization**: Neural Thompson Sampling with diagonal approximation
- **Context**: Can use transformer hidden states (in full implementation) or embeddings (simplified)

## Key Components

1. **NeuralTSDiag**: Neural bandit algorithm that uses a neural network surrogate
2. **INSTINCT**: Main optimizer class that integrates with the scoring function interface

## Usage

```python
from INSTINCT import INSTINCT
from transformers import GPT2Tokenizer, GPT2Model

# Setup embeddings
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
embedding_matrix = model.wte.weight.detach().numpy()
vocab_tokens = list(tokenizer.get_vocab().keys())

# Create scoring function
def score_instruction(instruction: str) -> float:
    # Evaluate instruction on your task
    return accuracy

# Initialize INSTINCT
instinct = INSTINCT(
    scoring_func=score_instruction,
    embedding_matrix=embedding_matrix,
    token_list=vocab_tokens,
    n_tokens=5,
    latent_dim=50,
    lambda_reg=0.1,
    nu=0.1
)

# Optimize
best_instruction, best_score = instinct.optimize(
    iterations=10,
    init_points=5
)
```

## Test Script

Run the test script with MATH500 dataset:

```bash
python test_instinct.py --samples 10 --iterations 10 --init_points 5
```

With API:
```bash
python test_instinct.py --use_api --api_model "meta-llama/llama-3.2-3b-instruct:free" --samples 10 --iterations 10
```

## Parameters

- `lambda_reg`: Regularization parameter (default: 0.1)
- `nu`: Exploration parameter (default: 0.1)
- `local_training_iter`: Neural network training iterations (default: 30)
- `n_domain`: Number of candidate points (default: 1000)
- `n_eval`: Points to evaluate per iteration (default: 100)

## Dependencies

- `torch>=2.0.0`
- `botorch>=0.9.0` (for SobolEngine)
- `backpack-for-pytorch>=1.6.0` (optional, for efficient gradients)

## Reference

Lin, X., Wu, Z., Dai, Z., et al. (2024). Use Your INSTINCT: INSTruction optimization for LLMs usIng Neural bandits Coupled with Transformers. ICML 2024.

