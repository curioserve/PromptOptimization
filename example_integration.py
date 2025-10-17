"""
Example integration of Soft Prompt Pool with InstructZero.

This script shows how to integrate the soft prompt pool system with the existing
InstructZero pipeline for efficient prompt optimization.
"""

import numpy as np
from typing import Tuple, List
from soft_prompt_pool import SoftPromptPool, Arm
from InstructZero.InstructZero.experiments.bai import ArmState, sequential_halving


def create_instructzero_evaluation_function(
    task_name: str = "math500",
    model_name: str = "vicuna",
    cache_dir: str = None
) -> callable:
    """
    Create an evaluation function that integrates with InstructZero.
    
    This is a placeholder that shows how to integrate with the actual InstructZero
    evaluation pipeline. In practice, you would:
    1. Use the soft prompt vector to generate an instruction
    2. Evaluate the instruction on the target task
    3. Return the score and generated instruction
    
    Args:
        task_name: Name of the task to evaluate on
        model_name: Name of the model to use
        cache_dir: Directory for caching results
        
    Returns:
        Evaluation function that takes a vector p and returns (score, instruction)
    """
    
    def evaluate_fn(p: np.ndarray) -> Tuple[float, str]:
        """
        Evaluate a soft prompt vector using InstructZero pipeline.
        
        Args:
            p: Soft prompt vector p ∈ R^d
            
        Returns:
            Tuple of (score, instruction_text)
        """
        # TODO: Replace this with actual InstructZero integration
        # This is a placeholder that simulates the evaluation process
        
        # Simulate instruction generation from soft prompt
        # In practice, this would use your instruction generation model g(p)
        instruction = f"Solve this {task_name} problem step by step. " \
                     f"Show your reasoning clearly and provide the final answer."
        
        # Simulate task evaluation
        # In practice, this would use your evaluation function f(instruction, task_data)
        # For now, we'll use a realistic score function based on vector properties
        
        # Create a realistic score based on vector characteristics
        norm = np.linalg.norm(p)
        sparsity = np.sum(np.abs(p) < 0.1) / len(p)
        
        # Simulate different performance patterns
        if norm > 0.8 and sparsity < 0.3:
            # High norm, low sparsity -> good performance
            base_score = 0.7 + 0.2 * np.random.random()
        elif norm < 0.3 and sparsity > 0.7:
            # Low norm, high sparsity -> moderate performance
            base_score = 0.4 + 0.3 * np.random.random()
        else:
            # Mixed characteristics -> variable performance
            base_score = 0.3 + 0.4 * np.random.random()
        
        # Add some task-specific variation
        if task_name == "math500":
            # Math problems might favor certain vector patterns
            if np.sum(p[:len(p)//2]) > 0.5:
                base_score += 0.1
        
        # Ensure score is in [0, 1] range
        score = max(0.0, min(1.0, base_score))
        
        return float(score), instruction
    
    return evaluate_fn


def convert_arms_to_armstates(arms: List[Arm]) -> List[ArmState]:
    """
    Convert Soft Prompt Pool Arms to BAI ArmStates.
    
    Args:
        arms: List of Arm objects from soft prompt pool
        
    Returns:
        List of ArmState objects for BAI algorithms
    """
    armstates = []
    for arm in arms:
        armstate = ArmState(
            idx=arm.idx,
            vector=arm.p,
            key=arm.key,
            metadata=arm.metadata
        )
        armstates.append(armstate)
    return armstates


def run_instructzero_with_pool(
    task_name: str = "math500",
    model_name: str = "vicuna",
    K_init: int = 1000,
    d: int = 10,
    K_active: int = 100,
    M_warmup: int = 200,
    bai_budget: int = 80,
    bai_batch_size: int = 4,
    seed: int = 42
):
    """
    Run the complete InstructZero pipeline with soft prompt pool.
    
    This function demonstrates the complete workflow:
    1. Build a large pool of soft prompts
    2. Screen to select active arms
    3. Use BAI to find the best arm
    
    Args:
        task_name: Name of the task to optimize for
        model_name: Name of the model to use
        K_init: Initial pool size
        d: Dimension of soft prompt vectors
        K_active: Number of active arms after screening
        M_warmup: Number of warm-up evaluations for screening
        bai_budget: Budget for BAI algorithm
        bai_batch_size: Batch size for BAI evaluations
        seed: Random seed for reproducibility
    """
    print(f"Running InstructZero with Soft Prompt Pool")
    print(f"Task: {task_name}, Model: {model_name}")
    print(f"Pool: {K_init} → {K_active} → 1 (best arm)")
    print("=" * 60)
    
    # Step 1: Build soft prompt pool
    print("Step 1: Building soft prompt pool...")
    pool_manager = SoftPromptPool()
    pool = pool_manager.build_pool(K_init=K_init, d=d, sampler="lhs", seed=seed)
    print(f"  Generated {len(pool)} soft prompt arms")
    
    # Step 2: Set up evaluation function
    print("\nStep 2: Setting up evaluation function...")
    evaluate_fn = create_instructzero_evaluation_function(task_name, model_name)
    pool_manager.set_evaluation_function(evaluate_fn)
    
    # Step 3: Screen the pool
    print("\nStep 3: Screening pool...")
    active_arms = pool_manager.screen_pool(
        pool,
        K_active=K_active,
        M_warmup=M_warmup,
        surrogate_type="mlp",
        seed=seed
    )
    print(f"  Selected {len(active_arms)} active arms")
    
    # Step 4: Convert to ArmStates for BAI
    print("\nStep 4: Converting to BAI format...")
    armstates = convert_arms_to_armstates(active_arms)
    print(f"  Converted {len(armstates)} arms to ArmState format")
    
    # Step 5: Run BAI
    print(f"\nStep 5: Running BAI with budget {bai_budget}...")
    
    def evaluate_armstate(armstate: ArmState) -> dict:
        """Evaluate an ArmState using the pool manager."""
        score, instruction = pool_manager.evaluate_p(armstate.vector)
        armstate.update(score, instruction)
        return {
            "score": score,
            "instruction": instruction,
            "arm_idx": armstate.idx
        }
    
    def logger(msg: str):
        """Simple logger for BAI."""
        print(f"  [BAI] {msg}")
    
    best_arm, history = sequential_halving(
        arms=armstates,
        evaluate_arm=evaluate_armstate,
        total_budget=bai_budget,
        batch_size=bai_batch_size,
        logger=logger
    )
    
    # Step 6: Results
    print(f"\nStep 6: Results")
    print(f"  Best arm index: {best_arm.idx}")
    print(f"  Best arm score: {best_arm.mean:.4f}")
    print(f"  Best arm pulls: {best_arm.pulls}")
    if best_arm.instruction:
        print(f"  Best instruction: {best_arm.instruction[:100]}...")
    
    # Summary statistics
    total_evaluations = sum(arm.pulls for arm in armstates)
    print(f"\nSummary:")
    print(f"  Total evaluations: {total_evaluations}")
    print(f"  Pool efficiency: {len(active_arms) / len(pool) * 100:.1f}% of pool used")
    print(f"  Evaluation efficiency: {total_evaluations / len(pool) * 100:.1f}% of pool evaluated")
    
    return best_arm, history, pool_manager


def main():
    """Run the example integration."""
    print("InstructZero + Soft Prompt Pool Integration Example")
    print("=" * 60)
    
    # Run with different configurations
    configs = [
        {
            "task_name": "math500",
            "K_init": 500,
            "K_active": 50,
            "M_warmup": 100,
            "bai_budget": 40
        },
        {
            "task_name": "translation_en-de",
            "K_init": 800,
            "K_active": 80,
            "M_warmup": 150,
            "bai_budget": 60
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*20} Configuration {i+1} {'='*20}")
        try:
            best_arm, history, pool_manager = run_instructzero_with_pool(**config)
            
            # Show cache statistics
            stats = pool_manager.get_cache_stats()
            print(f"\nCache statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"Configuration {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Integration example completed!")


if __name__ == "__main__":
    main()
