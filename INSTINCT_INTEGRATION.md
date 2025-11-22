# INSTINCT Integration Summary

## Overview

Successfully integrated **INSTINCT** (INSTruction optimization usIng Neural bandits Coupled with Transformers) from the paper "Use Your INSTINCT: INSTruction optimization for LLMs usIng Neural bandits Coupled with Transformers [ICML 2024]" into the prompt optimization project.

## What Was Added

### 1. Core Implementation (`INSTINCT/instinct.py`)

- **NeuralTSDiag**: Neural bandit algorithm that uses a neural network surrogate instead of Gaussian Process
- **INSTINCT**: Main optimizer class that follows the same interface as BOInG
- Simplified implementation that works with GPT-2 embeddings (similar to BOInG)
- Supports both local models and API-based evaluation

### 2. Test Script (`test_instinct.py`)

- Similar structure to `test_boing.py`
- Supports MATH500 dataset evaluation
- Works with both local models and API (OpenAI/OpenRouter)
- Optional instruction generation via LLM

### 3. Dependencies Updated

Added to `requirements.txt`:
- `torch>=2.0.0`
- `botorch>=0.9.0` (for SobolEngine)
- `backpack-for-pytorch>=1.6.0` (optional, for efficient gradient computation)

## Key Features

### Differences from BOInG

| Feature | BOInG | INSTINCT |
|---------|-------|----------|
| Surrogate Model | Gaussian Process | Neural Network |
| Optimization | UCB with GP | Neural Thompson Sampling |
| Context | Embeddings | Embeddings (simplified) or Transformer hidden states (full) |
| Training | No training needed | Local training iterations |

### Architecture

```
Latent Vector (50D)
    ↓ [Random Projection]
Embedding Space (768D)
    ↓ [Nearest Neighbor]
Discrete Tokens
    ↓ [Optional: LLM expansion]
Instruction
    ↓ [Scoring Function]
Score
    ↓ [Neural Bandit Update]
Next Candidate Selection
```

## Usage

### Basic Usage

```python
from INSTINCT import INSTINCT
from transformers import GPT2Tokenizer, GPT2Model

# Setup
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
embedding_matrix = model.wte.weight.detach().numpy()
vocab_tokens = list(tokenizer.get_vocab().keys())

# Scoring function
def score_instruction(instruction: str) -> float:
    # Your evaluation logic
    return accuracy

# Initialize
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

### Command Line

```bash
# Basic test
python test_instinct.py --samples 10 --iterations 10 --init_points 5

# With API
python test_instinct.py --use_api \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 10 --iterations 10

# With instruction generator
python test_instinct.py --use_api --use_generator \
    --api_model "gpt-3.5-turbo" \
    --samples 10 --iterations 10
```

## Parameters

- **lambda_reg** (default: 0.1): Regularization parameter for neural bandit
- **nu** (default: 0.1): Exploration parameter
- **local_training_iter** (default: 30): Neural network training iterations per update
- **n_domain** (default: 1000): Number of candidate points in domain
- **n_eval** (default: 100): Points to evaluate at each iteration

## Files Created

1. `INSTINCT/__init__.py` - Package initialization
2. `INSTINCT/instinct.py` - Core implementation
3. `INSTINCT/README.md` - Documentation
4. `INSTINCT/LlamaForMLPRegression.py` - Original neural bandit components (for reference)
5. `test_instinct.py` - Test script
6. `INSTINCT_INTEGRATION.md` - This file

## Next Steps

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the implementation**:
   ```bash
   python test_instinct.py --samples 5 --iterations 5 --init_points 3
   ```

3. **Compare with BOInG**:
   - Run both methods on the same task
   - Compare convergence speed and final performance
   - Analyze computational costs

## Notes

- The current implementation uses embeddings directly as context (simplified approach)
- The full INSTINCT paper uses transformer hidden states, which would require loading a transformer model (e.g., Vicuna)
- `backpack-for-pytorch` is optional but recommended for efficient gradient computation
- The implementation gracefully falls back to a simpler gradient computation if backpack is not available

## Reference

Lin, X., Wu, Z., Dai, Z., Hu, W., Shu, Y., Ng, S. K., Jaillet, P., & Low, B. K. H. (2024). Use Your INSTINCT: INSTruction optimization for LLMs usIng Neural bandits Coupled with Transformers. ICML 2024.

Original repository: https://github.com/xqlin98/INSTINCT

