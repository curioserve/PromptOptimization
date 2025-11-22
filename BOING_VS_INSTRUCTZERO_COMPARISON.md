# BOInG vs InstructZero: Comprehensive Comparison and Reporting Guide

## Table of Contents
1. [Method Overview](#method-overview)
2. [Key Differences](#key-differences)
3. [Experimental Comparison Setup](#experimental-comparison-setup)
4. [How to Generate Good Prompts](#how-to-generate-good-prompts)
5. [Reporting Guidelines](#reporting-guidelines)
6. [Code Examples](#code-examples)

---

## 1. Method Overview

### BOInG (Bayesian Optimization for Instruction Generation)

**Core Approach:**
- Optimizes in **low-dimensional latent space** (50D default)
- Uses **random projection** to map latent vectors to token embeddings
- Converts to **discrete tokens** via nearest neighbor in embedding space
- Uses **Gaussian Process** with **UCB acquisition function**
- **Direct token optimization**: Seed tokens → instruction (optional LLM expansion)

**Key Innovation:**
- Works directly with token embeddings (GPT-2)
- Simple random projection (no training needed)
- Penalized UCB balances exploration/exploitation

### InstructZero

**Core Approach:**
- Optimizes **soft prompt embeddings** in low-dimensional space (10D default)
- Uses **learned linear projection** to map latent to embedding space
- Generates instructions via **open-source LLM** (Vicuna, WizardLM, etc.)
- Uses **Gaussian Process** with **instruction-coupled kernel**
- **Soft prompt → LLM generation**: Latent → soft embeddings → LLM generates instruction

**Key Innovation:**
- Uses open-source LLM as instruction generator
- Instruction-coupled kernel models instruction similarity
- Two-stage: soft prompt optimization → instruction generation

---

## 2. Key Differences

### 2.1 Architecture Comparison

| Aspect | BOInG | InstructZero |
|--------|-------|--------------|
| **Latent Dimension** | 50 (default) | 10 (default) |
| **Projection** | Random matrix (fixed) | Learned linear layer |
| **Token Mapping** | Nearest neighbor in embedding space | Soft embeddings → LLM generation |
| **Instruction Generation** | Optional (tokens or LLM expansion) | Always via open-source LLM |
| **Search Space** | Token vocabulary (discrete) | Continuous embedding space |
| **GP Kernel** | Matern 5/2 | Instruction-coupled kernel |

### 2.2 Methodological Differences

#### **BOInG:**
```
Latent Vector (50D)
    ↓ [Random Projection Φ]
Embedding Space (768D)
    ↓ [Nearest Neighbor]
Discrete Tokens
    ↓ [Optional: LLM expansion]
Instruction
    ↓ [Scoring Function]
Score
```

#### **InstructZero:**
```
Latent Vector (10D)
    ↓ [Learned Linear Projection A]
Soft Prompt Embeddings
    ↓ [Concatenate with input]
Open-Source LLM (Vicuna/WizardLM)
    ↓ [Generate]
Instruction
    ↓ [Scoring Function]
Score
```

### 2.3 Advantages and Limitations

#### **BOInG Advantages:**
✅ **Simplicity**: No need for open-source LLM  
✅ **Flexibility**: Can work with any embedding model  
✅ **Direct control**: Works directly with tokens  
✅ **Lower computational cost**: No LLM generation step  
✅ **Interpretability**: Seed tokens are directly visible  

#### **BOInG Limitations:**
❌ **Limited expressiveness**: Fixed number of seed tokens  
❌ **Approximate mapping**: Nearest neighbor is not exact  
❌ **No semantic generation**: Can't generate novel instructions beyond vocabulary  

#### **InstructZero Advantages:**
✅ **Rich instruction generation**: LLM can generate complex instructions  
✅ **Semantic understanding**: LLM understands instruction semantics  
✅ **Instruction-coupled kernel**: Models instruction similarity better  
✅ **Proven effectiveness**: Published in ICML 2024  

#### **InstructZero Limitations:**
❌ **Requires open-source LLM**: Needs Vicuna/WizardLM loaded locally  
❌ **Higher computational cost**: LLM generation for each evaluation  
❌ **More complex setup**: Requires model loading and configuration  
❌ **Less interpretable**: Soft prompts are not directly readable  

---

## 3. Experimental Comparison Setup

### 3.1 Fair Comparison Criteria

To fairly compare BOInG and InstructZero, ensure:

1. **Same Task**: Use identical evaluation dataset (MATH500)
2. **Same Budget**: Same number of function evaluations
3. **Same Scoring Function**: Identical evaluation methodology
4. **Same Initialization**: Same random seed for reproducibility
5. **Same Hardware**: Run on same machine/GPU for timing

### 3.2 Experimental Design

#### **Metrics to Report:**

1. **Performance Metrics:**
   - Final best accuracy
   - Convergence speed (accuracy vs iterations)
   - Sample efficiency (accuracy vs number of evaluations)

2. **Computational Metrics:**
   - Total runtime
   - API calls (if using API)
   - Memory usage
   - Cost (if using paid APIs)

3. **Quality Metrics:**
   - Instruction quality (human evaluation)
   - Instruction diversity
   - Instruction interpretability

#### **Hyperparameters to Compare:**

| Parameter | BOInG | InstructZero |
|-----------|-------|--------------|
| Latent dim | 50 | 10 |
| Init points | 5 | Varies |
| Iterations | 10 | Varies |
| n_tokens/n_prompt_tokens | 5 | 3-10 |
| Beta (exploration) | 2.0 | Varies |
| Lambda (penalty) | 10.0 | N/A |

### 3.3 Comparison Script Structure

```python
# comparison_experiment.py

def run_comparison_experiment():
    """
    Fair comparison between BOInG and InstructZero
    """
    # 1. Setup
    task = "MATH500"
    num_samples = 10  # Evaluation samples
    budget = 15  # Total evaluations (init + iterations)
    
    # 2. Run BOInG
    boing_results = run_boing(
        init_points=5,
        iterations=10,
        num_samples=num_samples
    )
    
    # 3. Run InstructZero
    instructzero_results = run_instructzero(
        intrinsic_dim=10,
        n_prompt_tokens=5,
        num_samples=num_samples,
        budget=budget
    )
    
    # 4. Compare
    compare_results(boing_results, instructzero_results)
    
    # 5. Report
    generate_report(boing_results, instructzero_results)
```

---

## 4. How to Generate Good Prompts

### 4.1 BOInG Prompt Generation Strategy

#### **Strategy 1: Direct Token Usage**
```python
# Use seed tokens directly as instruction
boing = BOInG(
    scoring_func=scoring_func,
    embedding_matrix=embeddings,
    token_list=tokens,
    generator_func=None,  # No expansion
    n_tokens=5
)
```
**Best for:** Simple, concise instructions

#### **Strategy 2: LLM Expansion**
```python
# Expand seed tokens to full instruction
def expand_instruction(seed_tokens):
    prompt = f"Given: {seed_tokens}\nGenerate instruction:"
    return llm.generate(prompt)

boing = BOInG(
    scoring_func=scoring_func,
    embedding_matrix=embeddings,
    token_list=tokens,
    generator_func=expand_instruction,
    n_tokens=5
)
```
**Best for:** More expressive, natural instructions

#### **Tips for Better Prompts:**
1. **Increase n_tokens**: More tokens = more expressive (but larger search space)
2. **Tune lambda_penalty**: Lower = more exploration, Higher = stay near valid tokens
3. **Increase iterations**: More iterations = better optimization
4. **Use diverse init_points**: Better initial coverage

### 4.2 InstructZero Prompt Generation Strategy

#### **Setup:**
```python
# Requires open-source LLM
model_forward_api = LMForwardAPI(
    model_name="vicuna",  # or "wizardlm"
    eval_data=eval_data,
    init_prompt=None,
    init_qa=init_qa,
    random_proj=random_proj,
    intrinsic_dim=10,
    n_prompt_tokens=5
)
```

#### **Tips for Better Prompts:**
1. **Tune intrinsic_dim**: Higher = more capacity (default: 10)
2. **Adjust n_prompt_tokens**: 3-10 tokens (more = more expressive)
3. **Use instruction-coupled kernel**: Better models instruction similarity
4. **Provide good init_qa**: Better initialization helps

### 4.3 General Best Practices

#### **For Both Methods:**

1. **Start with Good Initialization:**
   - Use domain-specific seed tokens
   - Provide few-shot examples if possible
   - Use prior knowledge about the task

2. **Tune Hyperparameters:**
   - Latent dimension: Balance capacity vs efficiency
   - Exploration (beta): Higher for complex tasks
   - Number of iterations: More for better results

3. **Use Appropriate Scoring Function:**
   - Test on representative samples
   - Use consistent evaluation methodology
   - Consider multiple metrics

4. **Iterative Refinement:**
   - Start with small budget to explore
   - Increase budget for final optimization
   - Analyze best instructions for insights

---

## 5. Reporting Guidelines

### 5.1 Report Structure

#### **1. Introduction**
- Problem statement
- Motivation for comparison
- Research questions

#### **2. Methods**
- BOInG description
- InstructZero description
- Implementation details

#### **3. Experimental Setup**
- Dataset description
- Evaluation metrics
- Hyperparameters
- Hardware/software

#### **4. Results**
- Performance comparison
- Computational comparison
- Qualitative analysis
- Statistical significance

#### **5. Discussion**
- Strengths/weaknesses of each method
- When to use which method
- Future improvements

#### **6. Conclusion**
- Summary of findings
- Recommendations

### 5.2 Essential Tables

#### **Table 1: Performance Comparison**
| Method | Best Accuracy | Avg Accuracy | Std Dev | Convergence (iterations) |
|--------|---------------|--------------|---------|--------------------------|
| BOInG | 0.75 | 0.68 | 0.05 | 8 |
| InstructZero | 0.78 | 0.72 | 0.04 | 6 |
| Baseline | 0.60 | 0.58 | 0.03 | - |

#### **Table 2: Computational Comparison**
| Method | Total Time | API Calls | Memory (GB) | Cost ($) |
|--------|------------|-----------|-------------|----------|
| BOInG | 45 min | 150 | 2 | 0.50 |
| InstructZero | 120 min | 150 | 12 | 0.50 |

#### **Table 3: Hyperparameters**
| Parameter | BOInG | InstructZero |
|-----------|-------|--------------|
| Latent dim | 50 | 10 |
| Init points | 5 | 5 |
| Iterations | 10 | 10 |
| n_tokens | 5 | 5 |

### 5.3 Essential Figures

1. **Convergence Plot**: Accuracy vs Iterations
2. **Sample Efficiency**: Accuracy vs Number of Evaluations
3. **Instruction Quality**: Examples of generated instructions
4. **Computational Cost**: Time/Cost vs Performance

### 5.4 Statistical Analysis

```python
# Example statistical comparison
from scipy import stats

# Run multiple trials
boing_scores = [0.75, 0.73, 0.76, 0.74, 0.75]
instructzero_scores = [0.78, 0.77, 0.79, 0.76, 0.78]

# T-test
t_stat, p_value = stats.ttest_ind(boing_scores, instructzero_scores)
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")

# Effect size (Cohen's d)
cohens_d = (np.mean(instructzero_scores) - np.mean(boing_scores)) / \
           np.sqrt((np.var(boing_scores) + np.var(instructzero_scores)) / 2)
print(f"Cohen's d: {cohens_d:.3f}")
```

### 5.5 Qualitative Analysis

#### **Instruction Examples:**

**BOInG Best:**
```
"solve step by step carefully"
"think through each problem methodically"
"show all work clearly"
```

**InstructZero Best:**
```
"Please solve the following mathematical problem step by step, showing all your work and reasoning clearly."
"Think carefully about each step of the problem and explain your reasoning."
"Solve the problem systematically, breaking it down into smaller steps."
```

**Analysis:**
- BOInG: More concise, token-based
- InstructZero: More natural, LLM-generated

---

## 6. Code Examples

### 6.1 BOInG Implementation

```python
from BoIng import BOInG
from test_boing import create_scoring_function_api, setup_gpt2_embeddings

# Setup
embeddings, tokens, _ = setup_gpt2_embeddings()
scoring_func = create_scoring_function_api(
    api_key="your-key",
    model_name="meta-llama/llama-3.2-3b-instruct:free",
    num_samples=10
)

# Initialize
boing = BOInG(
    scoring_func=scoring_func,
    embedding_matrix=embeddings,
    token_list=tokens,
    n_tokens=5,
    latent_dim=50,
    beta=2.0,
    lambda_penalty=10.0
)

# Optimize
best_instruction, best_score = boing.optimize(
    iterations=10,
    init_points=5
)
```

### 6.2 InstructZero Implementation

```python
from InstructZero.InstructZero.experiments.run_instructzero import LMForwardAPI, run

# Setup (requires more configuration)
args = {
    'task': 'math500',
    'model_name': 'vicuna',
    'HF_cache_dir': './vicuna-13b',
    'intrinsic_dim': 10,
    'n_prompt_tokens': 5,
    'random_proj': random_proj_matrix
}

# Run
results = run(args)
best_instruction = results['best_instruction']
best_score = results['best_score']
```

### 6.3 Comparison Script

```python
import numpy as np
import time
from collections import defaultdict

def compare_methods(num_trials=5):
    """
    Compare BOInG and InstructZero across multiple trials
    """
    results = {
        'boing': {'scores': [], 'times': [], 'instructions': []},
        'instructzero': {'scores': [], 'times': [], 'instructions': []}
    }
    
    for trial in range(num_trials):
        # Run BOInG
        start = time.time()
        boing_instruction, boing_score = run_boing()
        boing_time = time.time() - start
        
        results['boing']['scores'].append(boing_score)
        results['boing']['times'].append(boing_time)
        results['boing']['instructions'].append(boing_instruction)
        
        # Run InstructZero
        start = time.time()
        iz_instruction, iz_score = run_instructzero()
        iz_time = time.time() - start
        
        results['instructzero']['scores'].append(iz_score)
        results['instructzero']['times'].append(iz_time)
        results['instructzero']['instructions'].append(iz_instruction)
    
    # Generate report
    print("="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nBOInG:")
    print(f"  Mean Accuracy: {np.mean(results['boing']['scores']):.3f} ± {np.std(results['boing']['scores']):.3f}")
    print(f"  Mean Time: {np.mean(results['boing']['times']):.2f}s")
    
    print(f"\nInstructZero:")
    print(f"  Mean Accuracy: {np.mean(results['instructzero']['scores']):.3f} ± {np.std(results['instructzero']['scores']):.3f}")
    print(f"  Mean Time: {np.mean(results['instructzero']['times']):.2f}s")
    
    return results
```

---

## 7. Recommendations

### When to Use BOInG:
- ✅ Need simple, fast optimization
- ✅ Don't have access to open-source LLMs
- ✅ Want interpretable seed tokens
- ✅ Limited computational resources
- ✅ Working with API-based models

### When to Use InstructZero:
- ✅ Need rich, natural instructions
- ✅ Have access to open-source LLMs (Vicuna, WizardLM)
- ✅ Want proven, published method
- ✅ Need instruction-coupled kernel benefits
- ✅ Have sufficient computational resources

### Hybrid Approach:
Consider combining both:
1. Use BOInG for initial exploration (fast)
2. Use InstructZero for refinement (better instructions)
3. Or use BOInG seed tokens as initialization for InstructZero

---

## 8. Future Work

1. **Hybrid Methods**: Combine BOInG's efficiency with InstructZero's expressiveness
2. **Better Kernels**: Improve GP kernels for instruction optimization
3. **Multi-objective**: Optimize for accuracy, cost, and instruction length
4. **Transfer Learning**: Use prompts from similar tasks
5. **Automated Hyperparameter Tuning**: Auto-tune latent_dim, beta, etc.

---

## Summary

**BOInG** offers simplicity and efficiency, working directly with token embeddings and requiring minimal setup. **InstructZero** provides richer instruction generation through LLM-based expansion but requires more computational resources.

The choice depends on your specific needs:
- **Speed & Simplicity**: Choose BOInG
- **Instruction Quality & Expressiveness**: Choose InstructZero
- **Best of Both**: Consider hybrid approaches

For reporting, ensure fair comparison with same budget, same task, and statistical significance testing.


