# Complete Explanation of BOInG Method and Scoring Function

## Overview
    
**BOInG (Bayesian Optimization for Instruction Generation)** is a method that uses Bayesian Optimization to find the best instruction/prompt for a given task. It optimizes in a low-dimensional latent space and uses token embeddings to convert latent vectors back to actual tokens.

---

## Part 1: BOInG Method - Complete Breakdown

### 1.1 Core Concept

BOInG searches for optimal instructions by:
1. **Working in latent space**: Instead of searching over all possible token combinations (which is huge), it searches in a low-dimensional continuous space
2. **Using embeddings**: Maps latent vectors to actual tokens using GPT-2 embeddings
3. **Bayesian Optimization**: Uses a Gaussian Process to model the relationship between instructions and their performance scores
4. **Acquisition function**: Balances exploration (trying new areas) and exploitation (refining known good areas)

### 1.2 Key Components

#### **Initialization (`__init__`)**

```python
BOInG(scoring_func, embedding_matrix, token_list, ...)
```

**Parameters:**
- `scoring_func`: Function that takes an instruction string and returns a score (higher = better)
- `embedding_matrix`: GPT-2 token embeddings (shape: V × D, where V = vocab size, D = embedding dim)
- `token_list`: List of token strings corresponding to embedding rows
- `n_tokens`: Number of tokens in the seed prompt (default: 5)
- `latent_dim`: Dimensionality of latent space (default: 50, much smaller than D=768)
- `lambda_penalty`: Weight for distance penalty (default: 10.0)
- `beta`: Exploration parameter for UCB (default: 2.0)

**What happens:**
1. Creates a random projection matrix Φ (shape: D × latent_dim) to map latent vectors to embedding space
2. Precomputes embedding norms for efficient distance calculations
3. Initializes a Gaussian Process (GP) surrogate model with Matern kernel
4. Sets up storage for training data (X_train, y_train) and best results

#### **Latent to Tokens Conversion (`latent_to_tokens`)**

This is the **core mapping function** that converts continuous latent vectors to discrete tokens.

**Process:**
1. Takes a latent prompt: shape (n_tokens, latent_dim) or flattened
2. For each token position:
   - Projects latent vector to embedding space: `embed_vec = Φ · z[i]`
   - Finds nearest neighbor token in embedding space using cosine distance
   - Uses optimized distance calculation: `||e - embed_vec||² = ||e||² + ||embed_vec||² - 2·e·embed_vec`
3. Returns list of actual token strings

**Why this works:**
- Latent space is continuous and smooth
- Embedding space captures semantic relationships
- Nearest neighbor ensures we get valid tokens

#### **Evaluation (`evaluate_latent_prompt`)**

This function evaluates a latent prompt and updates the GP model.

**Steps:**
1. **Convert to tokens**: `latent_prompt` → token list
2. **Generate instruction**: 
   - If `generator_func` provided: seed tokens → full instruction via LLM
   - Otherwise: seed tokens used directly as instruction
3. **Score instruction**: Calls `scoring_func(instruction)` to get performance
4. **Update GP**: Adds (latent_vector, score) to training data and refits GP
5. **Track best**: Updates best instruction if score is higher

**Returns:** `(score, tokens, instruction)`

#### **Acquisition Function (`acquisition_value`)**

This determines which latent prompt to evaluate next. Uses **Penalized Upper Confidence Bound (UCB)**:

```
acq(p) = μ(p) + β·σ(p) - λ·g(p)
```

Where:
- `μ(p)`: GP mean prediction (expected score)
- `σ(p)`: GP standard deviation (uncertainty)
- `β`: Exploration weight (higher = more exploration)
- `g(p)`: Distance penalty (average distance to nearest token embeddings)
- `λ`: Penalty weight (higher = prefer prompts closer to known tokens)

**Interpretation:**
- `μ(p) + β·σ(p)`: UCB - favors high expected value OR high uncertainty
- `-λ·g(p)`: Penalty - discourages prompts far from valid token embeddings
- Higher acquisition value = more promising to evaluate

#### **Proposal (`propose_next_latent`)**

Finds the next latent prompt to evaluate by maximizing the acquisition function.

**Method:**
1. Samples `n_candidates` random latent vectors (default: 100)
2. Computes acquisition value for each
3. Returns the one with highest acquisition value

**Note:** This is a simple random search. Could be improved with gradient-based optimization.

#### **Optimization Loop (`optimize`)**

The main optimization procedure.

**Phase 1: Initial Random Exploration**
```python
for j in range(init_points):  # default: 5
    latent = random_uniform(-1, 1)
    evaluate_latent_prompt(latent)
```
- Evaluates random prompts to build initial GP model
- Provides diverse starting points

**Phase 2: Bayesian Optimization**
```python
for i in range(iterations):  # default: 10
    latent = propose_next_latent()  # Uses acquisition function
    evaluate_latent_prompt(latent)
```
- Uses acquisition function to intelligently select next prompts
- Balances exploration and exploitation

**Returns:** `(best_instruction, best_score)`

---

## Part 2: Scoring Function - Complete Breakdown

### 2.1 Purpose

The scoring function evaluates how well an instruction performs on the target task (MATH500 dataset). It's the **objective function** that BOInG tries to maximize.

### 2.2 API-Based Scoring Function (`create_scoring_function_api`)

#### **Setup Phase**

```python
create_scoring_function_api(api_key, model_name, num_samples, max_tokens, use_openrouter)
```

**What it does:**
1. **Sets up API client**: Creates OpenAI client with OpenRouter or OpenAI base URL
2. **Loads dataset**: Loads MATH500 dataset (cached to avoid reloading)
3. **Creates evaluator**: Creates MATH500Evaluator (only for dataset/answer extraction, not model loading)

#### **Scoring Function (`score_instruction`)**

The actual function returned that evaluates an instruction.

**Input:** `instruction: str` - The instruction to evaluate

**Process:**
```python
for each sample in MATH500 dataset:
    1. Format prompt: "{instruction}\n\nProblem: {problem}\nSolution:"
    2. Call API: Send prompt to LLM (OpenRouter/OpenAI)
    3. Extract answer: Parse model's response to get predicted answer
    4. Check correctness: Compare predicted vs ground truth
    5. Count correct: Increment if correct
```

**Output:** `accuracy: float` (0.0 to 1.0) - Fraction of correct answers

**Key Details:**
- **Prompt format**: Instruction + Problem + "Solution:" prompt
- **Temperature**: 0.0 (deterministic)
- **Max tokens**: Configurable (default: 1024)
- **Error handling**: Continues on errors, counts as incorrect
- **Answer extraction**: Uses evaluator's `extract_answer()` method (handles LaTeX, boxed answers, etc.)
- **Correctness check**: Uses evaluator's `check_correctness()` method (handles mathematical equivalence)

### 2.3 Local Model Scoring Function (`create_scoring_function`)

Similar to API version but uses a local model instead of API calls.

**Differences:**
- Uses `evaluator.pipe` (transformers pipeline) instead of API
- Sets up local model pipeline if not already set up
- Same evaluation logic otherwise

---

## Part 3: How They Work Together

### 3.1 Complete Flow

```
1. BOInG Initialization
   ├─ Load GPT-2 embeddings
   ├─ Create random projection matrix
   └─ Initialize GP model

2. For each optimization iteration:
   ├─ BOInG proposes latent vector (using acquisition function)
   ├─ Convert latent → tokens (via embeddings)
   ├─ Generate instruction (via generator_func or use tokens directly)
   ├─ Score instruction (via scoring_func):
   │   ├─ For each MATH500 problem:
   │   │   ├─ Format prompt with instruction
   │   │   ├─ Call LLM API
   │   │   ├─ Extract answer
   │   │   └─ Check correctness
   │   └─ Return accuracy (0.0-1.0)
   ├─ Update GP with (latent, score)
   └─ Track best instruction

3. Return best instruction found
```

### 3.2 Example Walkthrough

**Iteration 1:**
- BOInG proposes: `latent = [0.2, -0.5, 0.8, ...]` (50-dim vector)
- Converts to tokens: `["solve", "step", "by", "step", "carefully"]`
- Generates instruction: `"solve step by step carefully"` (or uses LLM to expand)
- Scoring function:
  - Problem 1: "solve step by step carefully\n\nProblem: 2+2=?\nSolution:" → Model: "4" → Correct ✓
  - Problem 2: "solve step by step carefully\n\nProblem: 3×5=?\nSolution:" → Model: "15" → Correct ✓
  - ... (evaluates all samples)
  - Returns: `score = 0.60` (60% accuracy)
- GP learns: `latent → 0.60`
- Best so far: `0.60`

**Iteration 2:**
- GP predicts: Higher scores might be in different region
- Acquisition function suggests: `latent = [0.8, 0.1, -0.3, ...]`
- Converts to tokens: `["think", "carefully", "show", "work", "clearly"]`
- Generates instruction: `"think carefully and show your work clearly"`
- Scoring function returns: `score = 0.75` (75% accuracy)
- GP updates: Now knows about two regions
- Best updated: `0.75`

**Continues...** until iterations complete or convergence.

---

## Part 4: Key Design Choices

### 4.1 Why Latent Space?

- **Dimensionality**: Token space is huge (50257^5 combinations for 5 tokens)
- **Continuity**: Latent space is continuous, enabling smooth optimization
- **Efficiency**: 50 dimensions vs millions of token combinations

### 4.2 Why Random Projection?

- **Simplicity**: Fixed random matrix, no training needed
- **Preservation**: Random projections preserve distances (Johnson-Lindenstrauss lemma)
- **Flexibility**: Can use any embedding space

### 4.3 Why Gaussian Process?

- **Uncertainty**: Provides uncertainty estimates (needed for UCB)
- **Data efficiency**: Works well with few observations
- **Smoothness**: Assumes smooth function (reasonable for instruction quality)

### 4.4 Why Penalized UCB?

- **Exploration**: UCB encourages exploring uncertain regions
- **Exploitation**: UCB also exploits high-mean regions
- **Validity**: Penalty ensures we stay near valid token embeddings

### 4.5 Why Accuracy as Score?

- **Direct metric**: Accuracy directly measures task performance
- **Interpretable**: Easy to understand (0.0 = 0%, 1.0 = 100%)
- **Task-specific**: Fits the MATH500 evaluation goal

---

## Part 5: Parameters and Tuning

### 5.1 BOInG Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_tokens` | 5 | More tokens = more expressive but larger search space |
| `latent_dim` | 50 | Higher = more capacity but slower optimization |
| `lambda_penalty` | 10.0 | Higher = stronger preference for valid tokens |
| `beta` | 2.0 | Higher = more exploration vs exploitation |
| `init_points` | 5 | More = better initial GP model but slower start |
| `iterations` | 10 | More = better optimization but more API calls |

### 5.2 Scoring Function Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_samples` | 10 | More samples = more accurate score but slower |
| `max_tokens` | 1024 | More tokens = longer responses but more cost |
| `model_name` | varies | Different models = different capabilities/costs |

---

## Part 6: Advantages and Limitations

### 6.1 Advantages

✅ **Efficient**: Works in low-dimensional space  
✅ **Flexible**: Can use any embedding model  
✅ **Data-efficient**: GP works with few observations  
✅ **Interpretable**: Returns actual instructions  
✅ **Task-agnostic**: Works for any task with a scoring function  

### 6.2 Limitations

❌ **Approximate**: Nearest neighbor mapping is not exact  
❌ **Limited expressiveness**: Fixed number of seed tokens  
❌ **Computational cost**: Each evaluation requires multiple API calls  
❌ **GP scalability**: GP becomes slow with many observations  
❌ **Local optima**: May get stuck in local optima  

---

## Part 7: Example Usage

```python
# 1. Create scoring function
scoring_func = create_scoring_function_api(
    api_key="your-key",
    model_name="meta-llama/llama-3.2-3b-instruct:free",
    num_samples=10
)

# 2. Initialize BOInG
boing = BOInG(
    scoring_func=scoring_func,
    embedding_matrix=gpt2_embeddings,
    token_list=gpt2_tokens,
    n_tokens=5,
    latent_dim=50
)

# 3. Optimize
best_instruction, best_score = boing.optimize(
    iterations=10,
    init_points=5
)

# Result: best_instruction = "solve step by step carefully"
#         best_score = 0.75 (75% accuracy)
```

---

## Summary

**BOInG** is a smart search algorithm that:
- Searches in a continuous latent space (efficient)
- Maps to discrete tokens via embeddings (valid)
- Uses GP to model instruction quality (data-efficient)
- Uses UCB to balance exploration/exploitation (effective)

**Scoring Function** evaluates instructions by:
- Testing on multiple problems from MATH500
- Using LLM API to generate solutions
- Computing accuracy as the score
- Returning a single number (0.0-1.0) for BOInG to optimize

Together, they form a complete system for automatically finding the best instruction for a given task!


