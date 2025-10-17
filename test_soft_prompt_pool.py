"""
Test script for the Soft Prompt Pool System.

This script demonstrates the three main components:
1. Building a 10k soft-prompt pool
2. Lazy evaluation with caching
3. Screening to select active arms

Run this script to verify the system works correctly.
"""

import numpy as np
import time
from typing import Tuple
from soft_prompt_pool import SoftPromptPool, build_pool, screen_pool, Arm


def create_realistic_evaluation_function(
    noise_level: float = 0.1,
    complexity_factor: float = 1.0
) -> callable:
    """
    Create a realistic evaluation function that simulates the InstructZero pipeline.
    
    This function simulates:
    1. Generating an instruction from a soft prompt vector
    2. Evaluating the instruction on a task
    3. Returning a score and the generated instruction text
    
    Args:
        noise_level: Amount of noise to add to the score
        complexity_factor: Controls the complexity of the score function
        
    Returns:
        Evaluation function that takes a vector p and returns (score, instruction)
    """
    def evaluate_fn(p: np.ndarray) -> Tuple[float, str]:
        # Simulate instruction generation complexity
        # Higher norm vectors tend to generate more complex instructions
        norm = np.linalg.norm(p)
        
        # Simulate a realistic score function with multiple peaks
        # This creates a challenging optimization landscape
        score = 0.0
        
        # Peak 1: Favor vectors with specific patterns
        if np.abs(p[0]) > 0.5 and np.abs(p[1]) < 0.3:
            score += 0.3
        
        # Peak 2: Favor vectors with certain combinations
        if np.sum(p[:3]) > 0.5 and np.sum(p[3:]) < 0.2:
            score += 0.4
        
        # Peak 3: Favor vectors with alternating signs
        alternating = np.sum(np.abs(np.diff(np.sign(p))))
        if alternating > len(p) * 0.6:
            score += 0.2
        
        # Add some smoothness based on vector properties
        score += 0.1 * np.exp(-norm / complexity_factor)
        
        # Add noise
        score += np.random.normal(0, noise_level)
        
        # Ensure score is in reasonable range
        score = max(0.0, min(1.0, score))
        
        # Generate a realistic instruction text
        instruction_parts = []
        if norm > 0.8:
            instruction_parts.append("Complex reasoning task:")
        elif norm > 0.4:
            instruction_parts.append("Moderate difficulty task:")
        else:
            instruction_parts.append("Simple task:")
        
        # Add some variation based on vector components
        if p[0] > 0:
            instruction_parts.append("Analyze the following")
        else:
            instruction_parts.append("Generate a response for")
        
        if p[1] > 0:
            instruction_parts.append("mathematical problem")
        else:
            instruction_parts.append("textual content")
        
        instruction = " ".join(instruction_parts) + f" (vector_norm={norm:.3f})"
        
        return float(score), instruction
    
    return evaluate_fn


def test_pool_building():
    """Test the pool building functionality."""
    print("Testing Pool Building")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        {"K_init": 100, "d": 5, "sampler": "lhs", "seed": 42},
        {"K_init": 50, "d": 10, "sampler": "uniform", "seed": 123},
        {"K_init": 200, "d": 3, "sampler": "lhs", "seed": 456},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        start_time = time.time()
        
        pool = build_pool(**config)
        
        end_time = time.time()
        print(f"  Generated {len(pool)} arms in {end_time - start_time:.3f} seconds")
        print(f"  First arm shape: {pool[0].p.shape}")
        print(f"  First arm vector: {pool[0].p}")
        print(f"  Vector range: [{pool[0].p.min():.3f}, {pool[0].p.max():.3f}]")
        
        # Verify properties
        assert len(pool) == config["K_init"], f"Expected {config['K_init']} arms, got {len(pool)}"
        assert pool[0].p.shape[0] == config["d"], f"Expected dimension {config['d']}, got {pool[0].p.shape[0]}"
        assert all(-1 <= p <= 1 for p in pool[0].p), "Vectors should be in [-1, 1] range"
        
        print("  ✓ Configuration passed all tests")


def test_lazy_evaluation():
    """Test the lazy evaluation with caching."""
    print("\n\nTesting Lazy Evaluation")
    print("=" * 50)
    
    # Create a small pool for testing
    pool = build_pool(K_init=20, d=5, sampler="lhs", seed=42)
    evaluate_fn = create_realistic_evaluation_function(noise_level=0.05)
    
    # Create pool manager
    pool_manager = SoftPromptPool()
    pool_manager.set_evaluation_function(evaluate_fn)
    
    # Test evaluation
    print("Evaluating first arm...")
    start_time = time.time()
    score1, instruction1 = pool_manager.evaluate_p(pool[0].p)
    eval_time1 = time.time() - start_time
    print(f"  Score: {score1:.4f}")
    print(f"  Instruction: {instruction1}")
    print(f"  Evaluation time: {eval_time1:.3f} seconds")
    
    # Test caching
    print("\nEvaluating same arm again (should use cache)...")
    start_time = time.time()
    score2, instruction2 = pool_manager.evaluate_p(pool[0].p)
    eval_time2 = time.time() - start_time
    print(f"  Score: {score2:.4f}")
    print(f"  Instruction: {instruction2}")
    print(f"  Evaluation time: {eval_time2:.3f} seconds")
    print(f"  Cache hit: {score1 == score2} (scores should be identical)")
    print(f"  Speedup: {eval_time1 / eval_time2:.1f}x faster")
    
    # Test multiple evaluations
    print(f"\nEvaluating {len(pool)} arms...")
    start_time = time.time()
    scores = []
    for arm in pool:
        score, _ = pool_manager.evaluate_p(arm.p)
        scores.append(score)
    total_time = time.time() - start_time
    
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Average time per arm: {total_time / len(pool):.3f} seconds")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Score mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Test cache statistics
    stats = pool_manager.get_cache_stats()
    print(f"\nCache statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_screening():
    """Test the screening functionality."""
    print("\n\nTesting Screening")
    print("=" * 50)
    
    # Create a larger pool for screening
    pool = build_pool(K_init=500, d=8, sampler="lhs", seed=42)
    evaluate_fn = create_realistic_evaluation_function(noise_level=0.1)
    
    # Create pool manager
    pool_manager = SoftPromptPool()
    pool_manager.set_evaluation_function(evaluate_fn)
    
    # Test different screening configurations
    configs = [
        {"K_active": 50, "M_warmup": 100, "surrogate_type": "linear"},
        {"K_active": 30, "M_warmup": 80, "surrogate_type": "rf"},
        {"K_active": 40, "M_warmup": 120, "surrogate_type": "mlp"},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nScreening Configuration {i+1}: {config}")
        start_time = time.time()
        
        active_arms = pool_manager.screen_pool(pool, **config)
        
        end_time = time.time()
        print(f"  Selected {len(active_arms)} active arms from {len(pool)} total")
        print(f"  Screening time: {end_time - start_time:.3f} seconds")
        
        # Verify selection
        assert len(active_arms) == config["K_active"], f"Expected {config['K_active']} active arms, got {len(active_arms)}"
        assert all(arm in pool for arm in active_arms), "All active arms should be from the original pool"
        
        # Evaluate the selected arms to see their actual performance
        print("  Evaluating selected arms...")
        selected_scores = []
        for arm in active_arms:
            score, _ = pool_manager.evaluate_p(arm.p)
            selected_scores.append(score)
        
        print(f"  Selected arms score range: [{min(selected_scores):.4f}, {max(selected_scores):.4f}]")
        print(f"  Selected arms mean score: {np.mean(selected_scores):.4f} ± {np.std(selected_scores):.4f}")
        
        print("  ✓ Configuration passed all tests")


def test_integration():
    """Test the complete integration workflow."""
    print("\n\nTesting Complete Integration")
    print("=" * 50)
    
    # Step 1: Build a large pool
    print("Step 1: Building 10k soft-prompt pool...")
    start_time = time.time()
    pool = build_pool(K_init=1000, d=10, sampler="lhs", seed=42)  # Using 1k for demo
    build_time = time.time() - start_time
    print(f"  Built {len(pool)} arms in {build_time:.3f} seconds")
    
    # Step 2: Set up evaluation
    print("\nStep 2: Setting up evaluation function...")
    evaluate_fn = create_realistic_evaluation_function(noise_level=0.15)
    pool_manager = SoftPromptPool()
    pool_manager.set_evaluation_function(evaluate_fn)
    
    # Step 3: Screen the pool
    print("\nStep 3: Screening pool to select active arms...")
    start_time = time.time()
    active_arms = pool_manager.screen_pool(
        pool, 
        K_active=100, 
        M_warmup=200, 
        surrogate_type="mlp"
    )
    screen_time = time.time() - start_time
    print(f"  Selected {len(active_arms)} active arms in {screen_time:.3f} seconds")
    
    # Step 4: Evaluate active arms
    print("\nStep 4: Evaluating active arms...")
    start_time = time.time()
    final_scores = []
    for i, arm in enumerate(active_arms):
        score, instruction = pool_manager.evaluate_p(arm.p)
        final_scores.append(score)
        if i < 3:  # Show first few examples
            print(f"  Arm {i+1}: score={score:.4f}, instruction='{instruction[:60]}...'")
    
    eval_time = time.time() - start_time
    print(f"  Evaluated {len(active_arms)} arms in {eval_time:.3f} seconds")
    print(f"  Final score range: [{min(final_scores):.4f}, {max(final_scores):.4f}]")
    print(f"  Best arm score: {max(final_scores):.4f}")
    
    # Summary
    total_time = build_time + screen_time + eval_time
    print(f"\nIntegration Summary:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Pool size: {len(pool)} arms")
    print(f"  Active arms: {len(active_arms)} arms")
    print(f"  Evaluation efficiency: {len(active_arms) / len(pool) * 100:.1f}% of arms evaluated")
    print(f"  Best performance: {max(final_scores):.4f}")
    
    # Cache statistics
    stats = pool_manager.get_cache_stats()
    print(f"\nFinal cache statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    """Run all tests."""
    print("Soft Prompt Pool System - Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        test_pool_building()
        test_lazy_evaluation()
        test_screening()
        test_integration()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("The Soft Prompt Pool System is ready for use.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
