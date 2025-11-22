# Prompt Optimization Framework

A comprehensive framework for optimizing instructions/prompts for Large Language Models (LLMs) using three state-of-the-art methods:

1. **BOInG** (Bayesian Optimization for Instruction Generation) - Uses Gaussian Process surrogate
2. **InstructZero** - Uses instruction-coupled kernel with open-source LLM
3. **INSTINCT** (INSTruction optimization usIng Neural bandits Coupled with Transformers) - Uses neural bandits with transformer representations

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Individual Methods](#running-individual-methods)
- [Comparing All Methods](#comparing-all-methods)
- [Method Details](#method-details)
- [Results](#results)

## Installation



### 1. Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate 
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys (for API-based evaluation)

```bash
# For OpenAI
export OPENAI_API_KEY=your_key_here

# For OpenRouter 
export OPENROUTER_API_KEY=your_key_here
```

**Note**: For InstructZero, you may need additional setup if using local models. See [InstructZero Setup](#instructzero-setup) below.

## Quick Start

### Compare All Three Methods

The easiest way to compare all methods is using the comparison script:

```bash
# Set API key first
export OPENROUTER_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here

# Run comparison with API
python compare_all_methods.py \
    --use_api \
    --use_openrouter \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 10 \
    --iterations 10 \
    --init_points 5
```

This will:
- Run BOInG, INSTINCT, and InstructZero with the same budget
- Compare their performance
- Generate a comparison report in `comparison_results/`

### Quick Test (Small Budget)

For a quick test with minimal API calls:

```bash
python compare_all_methods.py \
    --use_api \
    --use_openrouter \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 5 \
    --iterations 3 \
    --init_points 2
```

## Running Individual Methods

### 1. BOInG (Bayesian Optimization for Instruction Generation)

**Method**: Uses Gaussian Process with UCB acquisition function

```bash
# With API (OpenRouter)
python test_boing.py \
    --use_api \
    --use_openrouter \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 10 \
    --iterations 10 \
    --init_points 5

# With API (OpenAI)
python test_boing.py \
    --use_api \
    --api_model "gpt-3.5-turbo" \
    --samples 10 \
    --iterations 10 \
    --init_points 5

# With local model
python test_boing.py \
    --model "./gpt-oss-20b" \
    --samples 10 \
    --iterations 10 \
    --init_points 5

# With instruction generator (LLM expands seed tokens)
python test_boing.py \
    --use_api \
    --use_openrouter \
    --use_generator \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 10 \
    --iterations 10
```

**Key Parameters**:
- `--n_tokens`: Number of tokens in seed prompt (default: 5)
- `--latent_dim`: Latent dimension (default: 50)
- `--beta`: Exploration parameter (default: 2.0)
- `--use_generator`: Use LLM to generate instructions from seed tokens

### 2. INSTINCT (Neural Bandits)

**Method**: Uses neural network surrogate with Neural Thompson Sampling

```bash
# With API (OpenRouter)
python test_instinct.py \
    --use_api \
    --use_openrouter \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 10 \
    --iterations 10 \
    --init_points 5

# With API (OpenAI)
python test_instinct.py \
    --use_api \
    --api_model "gpt-3.5-turbo" \
    --samples 10 \
    --iterations 10 \
    --init_points 5

# With local model
python test_instinct.py \
    --model "./gpt-oss-20b" \
    --samples 10 \
    --iterations 10 \
    --init_points 5

# With instruction generator
python test_instinct.py \
    --use_api \
    --use_openrouter \
    --use_generator \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 10 \
    --iterations 10
```

**Key Parameters**:
- `--n_tokens`: Number of tokens in seed prompt (default: 5)
- `--latent_dim`: Latent dimension (default: 50)
- `--lambda_reg`: Regularization parameter (default: 0.1)
- `--nu`: Exploration parameter (default: 0.1)
- `--local_training_iter`: Neural network training iterations (default: 30)
- `--n_domain`: Number of candidate points (default: 1000)
- `--n_eval`: Points to evaluate per iteration (default: 100)

### 3. InstructZero

**Method**: Uses instruction-coupled kernel with open-source LLM

**Note**: InstructZero requires additional setup with an open-source LLM (e.g., Vicuna). For API-based evaluation, use the comparison script which handles this automatically.

```bash
# Using comparison script (recommended for API-based evaluation)
python compare_all_methods.py \
    --use_api \
    --use_openrouter \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 10 \
    --iterations 10 \
    --init_points 5

# Direct usage (requires LLM setup - see InstructZero/README.md)
cd InstructZero/InstructZero/experiments
bash run_instructzero.sh
```

**Setup Requirements**:
- Open-source LLM (e.g., Vicuna-13B, WizardLM)
- Model path configuration
- See `InstructZero/README.md` for detailed setup instructions

## Comparing All Methods

### Automated Comparison Script

The `compare_all_methods.py` script runs all three methods with the same budget and generates a comparison report:

```bash
# Set API key
export OPENROUTER_API_KEY=your_key_here

# Run comparison
python compare_all_methods.py \
    --use_api \
    --use_openrouter \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 10 \
    --iterations 10 \
    --init_points 5 \
    --output comparison_results

# With instruction generator
python compare_all_methods.py \
    --use_api \
    --use_openrouter \
    --use_generator \
    --api_model "meta-llama/llama-3.2-3b-instruct:free" \
    --samples 10 \
    --iterations 10 \
    --init_points 5
```

**Output**:
- Individual results for each method
- Comparison table with metrics
- Best instructions from each method
- JSON report with all details
- Text summary for quick viewing

Results are saved in `comparison_results/comparison_YYYYMMDD_HHMMSS/`:
- `results.json` - Full results in JSON format
- `summary.txt` - Human-readable summary

### Comparison Metrics

The comparison includes:
- **Best Score**: Highest accuracy achieved
- **Total Time**: Wall-clock time for optimization
- **Time per Evaluation**: Average time per function call
- **Number of Evaluations**: Function calls made
- **Best Instruction**: The optimized instruction text

### Example Output

```
================================================================================
COMPARISON RESULTS
================================================================================

Method          Best Score    Time (s)    Time/Eval    Evaluations
--------------------------------------------------------------------------------
BOInG           0.650         45.2        3.013        15
INSTINCT        0.680         52.1        3.473        15
InstructZero    0.670         38.5        2.567        15
--------------------------------------------------------------------------------

Winner: INSTINCT (Score: 0.680)
```

## Method Details

### BOInG

- **Surrogate**: Gaussian Process with Matern kernel
- **Acquisition**: Upper Confidence Bound (UCB)
- **Latent Space**: 50D (default)
- **Pros**: Simple, no training needed, fast
- **Cons**: Limited expressiveness of GP

### INSTINCT

- **Surrogate**: Neural Network (MLP)
- **Acquisition**: Neural Thompson Sampling (UCB style)
- **Latent Space**: 50D (default)
- **Pros**: More expressive than GP, can model complex functions
- **Cons**: Requires training iterations, slightly slower

### InstructZero

- **Surrogate**: Gaussian Process with instruction-coupled kernel
- **Acquisition**: Expected Improvement
- **Latent Space**: 10D (default)
- **Pros**: Uses LLM for instruction generation, instruction-aware kernel
- **Cons**: Requires open-source LLM setup, more complex

## Results

Results are saved in the `comparison_results/` directory (or specified `--output`):

```
comparison_results/
├── comparison_20240101_120000/
│   ├── results.json          # Full results (all methods)
│   └── summary.txt           # Human-readable summary
```

Individual method results are also available:
- `test_boing.py` outputs results to console
- `test_instinct.py` outputs results to console
- Both can be redirected to files: `python test_boing.py ... > boing_results.txt`

## Advanced Usage

### Custom Scoring Function

You can create custom scoring functions for different tasks:

```python
def my_scoring_function(instruction: str) -> float:
    # Your evaluation logic
    # Return a score (higher is better)
    return accuracy

# Use with any method
from BoIng import BOInG
boing = BOInG(scoring_func=my_scoring_function, ...)
```

### Using Instruction Generator

All methods support optional instruction generation from seed tokens:

```bash
python test_boing.py --use_api --use_generator \
    --api_model "gpt-3.5-turbo" \
    --samples 10
```

### Hyperparameter Tuning

Each method has different hyperparameters. Experiment with:

**BOInG**:
- `--beta`: Higher = more exploration
- `--latent_dim`: Higher = more expressive but slower

**INSTINCT**:
- `--lambda_reg`: Higher = more regularization
- `--nu`: Higher = more exploration
- `--local_training_iter`: More iterations = better fit but slower

**InstructZero**:
- `--intrinsic_dim`: Latent dimension (default: 10)
- `--n_prompt_tokens`: Number of soft prompt tokens

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **API key errors**: Set environment variables
   ```bash
   export OPENROUTER_API_KEY=your_key
   ```

3. **CUDA/GPU issues**: Methods will automatically use CPU if GPU unavailable

4. **InstructZero setup**: Requires additional LLM model setup. See below.

### InstructZero Setup

InstructZero requires an open-source LLM (e.g., Vicuna). To set up:

1. Download a compatible model (e.g., Vicuna-13B)
2. Set the model path in the configuration
3. Ensure sufficient GPU memory

For API-based evaluation, the comparison script handles this automatically.
=tation if available)


