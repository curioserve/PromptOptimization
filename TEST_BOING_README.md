# Testing BOInG with MATH500 Dataset

This guide explains how to test the BOInG (Bayesian Optimization for Instruction Generation) class with the MATH500 dataset.

## Overview

The `test_boing.py` script integrates:
1. **BOInG optimizer** - Uses GPT-2 embeddings to optimize instructions in a latent space
2. **MATH500 evaluator** - Evaluates instructions on the MATH500 math problem dataset
3. **Optional LLM generator** - Uses OpenAI API to generate instructions from seed prompts

## Prerequisites

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. (Optional) If you want to use the instruction generator, install OpenAI:
```bash
pip install openai
```

3. Set up your model for MATH500 evaluation (default: `./gpt-oss-20b`)

## Basic Usage

### Without Instruction Generator (Seed Prompts = Instructions)

This is the simplest mode - the seed tokens are used directly as instructions:

```bash
python test_boing.py \
    --model ./gpt-oss-20b \
    --samples 10 \
    --iterations 10 \
    --init_points 5
```

### With Instruction Generator (Requires OpenAI API Key)

If you want to use an LLM to generate instructions from seed prompts:

```bash
# Option 1: Pass API key as argument
python test_boing.py \
    --model ./gpt-oss-20b \
    --samples 10 \
    --iterations 10 \
    --init_points 5 \
    --use_generator \
    --api_key YOUR_OPENAI_API_KEY

# Option 2: Use environment variable
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
python test_boing.py \
    --model ./gpt-oss-20b \
    --samples 10 \
    --iterations 10 \
    --init_points 5 \
    --use_generator
```

## Command Line Arguments

- `--model`: Model ID for MATH500 evaluation (default: `./gpt-oss-20b`)
- `--samples`: Number of MATH500 samples to evaluate on (default: 10)
- `--iterations`: Number of BO iterations (default: 10)
- `--init_points`: Number of initial random evaluations (default: 5)
- `--n_tokens`: Number of tokens in seed prompt (default: 5)
- `--latent_dim`: Latent dimension for BO (default: 50)
- `--api_key`: OpenAI API key for instruction generation (optional)
- `--use_generator`: Use LLM to generate instructions from seed prompts
- `--max_tokens`: Max tokens for MATH500 evaluation (default: 1024)

## Example Output

```
================================================================================
BOInG + MATH500 Test
================================================================================
Model: ./gpt-oss-20b
MATH500 samples: 10
BO iterations: 10
Initial points: 5
Seed prompt tokens: 5
Use instruction generator: False
================================================================================

Loading GPT-2 tokenizer and model...
Loaded embeddings: 50257 tokens, 768 dimensions

Initializing MATH500 evaluator...

Creating scoring function...
Loading MATH500 dataset (first 10 samples)...
Loaded 10 samples

Initializing BOInG optimizer...

================================================================================
Starting BOInG optimization...
================================================================================
[Initial] Prompt 1: score=0.3000, tokens=['the', 'quick', 'brown', 'fox', 'jumps'], instruction="the quick brown fox jumps"
...
[Iteration 1] score=0.4000, tokens=['solve', 'step', 'by', 'step', 'carefully'], instruction="solve step by step carefully"
...
Best instruction found: "solve step by step carefully" (score=0.4000)

================================================================================
OPTIMIZATION COMPLETE
================================================================================
Best instruction: solve step by step carefully
Best score (accuracy): 0.4000
================================================================================
```

## How It Works

1. **Embedding Setup**: Loads GPT-2 tokenizer and model to get embeddings for the vocabulary
2. **Scoring Function**: Creates a function that evaluates an instruction by:
   - Running the MATH500 evaluator with the instruction
   - Computing accuracy across samples
   - Returning the accuracy as the score
3. **BOInG Optimization**: 
   - Starts with random seed prompts (initial points)
   - Uses Bayesian Optimization to find better prompts in latent space
   - Projects latent vectors to tokens via nearest neighbors
   - Optionally generates instructions from seed prompts using LLM
   - Evaluates each instruction on MATH500
4. **Results**: Returns the best instruction and its accuracy score

## Tips

- Start with a small number of samples (`--samples 10`) for faster testing
- Increase `--iterations` and `--init_points` for better results (but slower)
- The instruction generator is optional - you can test without it first
- The scoring function caches the dataset to avoid reloading

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
- **Model not found**: Check that your model path is correct (`--model`)
- **OpenAI API errors**: If using `--use_generator`, ensure your API key is valid
- **Memory issues**: Reduce `--samples` or `--n_tokens` if you run out of memory

