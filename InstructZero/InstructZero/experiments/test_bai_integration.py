#!/usr/bin/env python3
"""
Test script to verify BAI-FB integration in InstructZero.

This script tests the basic functionality of the BAI controller
without running the full evaluation pipeline.
"""

import sys
import os
import torch
import argparse
from unittest.mock import Mock, MagicMock

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bai import ArmState, sequential_halving, continuous_reject, cluster_bai_in_cluster, global_surrogate_elimination


def test_arm_state():
    """Test ArmState functionality."""
    print("Testing ArmState...")
    
    arm = ArmState(idx=0, vector=torch.randn(10), key=(0.1, 0.2, 0.3))
    
    # Test initial state
    assert arm.pulls == 0
    assert arm.mean == float("-inf")
    
    # Test updates
    arm.update(0.5, "test instruction")
    assert arm.pulls == 1
    assert arm.mean == 0.5
    assert arm.instruction == "test instruction"
    
    arm.update(0.7, "test instruction 2")
    assert arm.pulls == 2
    assert abs(arm.mean - 0.6) < 1e-6  # (0.5 + 0.7) / 2
    
    print("✓ ArmState tests passed")


def test_bai_methods():
    """Test BAI method functionality with mock evaluator."""
    print("Testing BAI methods...")
    
    # Create test arms
    arms = [
        ArmState(idx=i, vector=torch.randn(10), key=(i,)) 
        for i in range(8)
    ]
    
    # Mock evaluator that returns deterministic scores
    def mock_evaluate_arm(arm):
        score = 0.5 + (arm.idx * 0.1)  # Increasing scores
        arm.update(score, f"instruction_{arm.idx}")
        return {'score': score, 'instruction': f"instruction_{arm.idx}"}
    
    # Test Sequential Halving
    print("  Testing Sequential Halving...")
    best_arm, history = sequential_halving(
        arms.copy(), mock_evaluate_arm, total_budget=20, batch_size=2
    )
    assert best_arm.idx == 7  # Should be the arm with highest score
    assert len(history) > 0
    print("  ✓ Sequential Halving passed")
    
    # Test Continuous Reject
    print("  Testing Continuous Reject...")
    arms_cr = [
        ArmState(idx=i, vector=torch.randn(10), key=(i,)) 
        for i in range(8)
    ]
    best_arm_cr, history_cr = continuous_reject(
        arms_cr, mock_evaluate_arm, total_budget=20, batch_size=2, delta=0.1
    )
    assert best_arm_cr.idx == 7
    print("  ✓ Continuous Reject passed")
    
    # Test Cluster BAI (simplified)
    print("  Testing Cluster BAI...")
    arms_clst = [
        ArmState(idx=i, vector=torch.randn(10), key=(i,)) 
        for i in range(8)
    ]
    best_arm_clst, history_clst = cluster_bai_in_cluster(
        arms_clst, mock_evaluate_arm, total_budget=20, batch_size=2, 
        num_clusters=2, embedding_model=None
    )
    assert best_arm_clst.idx == 7
    print("  ✓ Cluster BAI passed")
    
    # Test Global Surrogate Elimination
    print("  Testing Global Surrogate Elimination...")
    arms_gse = [
        ArmState(idx=i, vector=torch.randn(10), key=(i,)) 
        for i in range(8)
    ]
    best_arm_gse, history_gse = global_surrogate_elimination(
        arms_gse, mock_evaluate_arm, total_budget=20, batch_size=2,
        surrogate_model='linear', embedding_model=None
    )
    assert best_arm_gse.idx == 7
    print("  ✓ Global Surrogate Elimination passed")
    
    print("✓ All BAI methods tests passed")


def test_arm_builders():
    """Test arm builder functionality."""
    print("Testing arm builders...")
    
    # Mock the required functions
    def _ensure_cpu_latent(latent):
        return latent.cpu()
    
    def _latent_key(latent):
        return tuple(round(float(v), 6) for v in latent.tolist())
    
    # Test random arm builder
    print("  Testing random arm builder...")
    import torch
    generator = torch.Generator().manual_seed(42)
    num_arms = 10
    intrinsic_dim = 5
    
    arm_map = {}
    while len(arm_map) < num_arms:
        remaining = num_arms - len(arm_map)
        samples = torch.rand((remaining, intrinsic_dim), generator=generator)
        for sample in samples:
            latent = _ensure_cpu_latent(sample * 2 - 1)
            key = _latent_key(latent)
            if key not in arm_map:
                arm_map[key] = ArmState(idx=len(arm_map), vector=latent.clone(), key=key)
    
    assert len(arm_map) == num_arms
    print("  ✓ Random arm builder passed")
    
    # Test LHS arm builder
    print("  Testing LHS arm builder...")
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=intrinsic_dim, seed=42)
        samples = sampler.random(n=num_arms)
        
        lhs_arm_map = {}
        for sample in samples:
            latent = _ensure_cpu_latent(torch.tensor(sample * 2 - 1, dtype=torch.float32))
            key = _latent_key(latent)
            if key not in lhs_arm_map:
                lhs_arm_map[key] = ArmState(idx=len(lhs_arm_map), vector=latent.clone(), key=key)
        
        assert len(lhs_arm_map) == num_arms
        print("  ✓ LHS arm builder passed")
    except ImportError:
        print("  ⚠ LHS arm builder skipped (scipy not available)")
    
    print("✓ Arm builders tests passed")


def test_args_parsing():
    """Test argument parsing for BAI options."""
    print("Testing argument parsing...")
    
    # Test with BAI arguments
    test_args = [
        '--selection', 'bai',
        '--bai_method', 'sh',
        '--bai_total_budget', '100',
        '--bai_batch_size', '4',
        '--bai_arm_builder', 'lhs',
        '--bai_num_clusters', '4',
        '--bai_surrogate_model', 'rf'
    ]
    
    # Mock sys.argv
    original_argv = sys.argv
    sys.argv = ['test_script.py'] + test_args
    
    try:
        from args import parse_args
        args = parse_args()
        
        assert args.selection == 'bai'
        assert args.bai_method == 'sh'
        assert args.bai_total_budget == 100
        assert args.bai_batch_size == 4
        assert args.bai_arm_builder == 'lhs'
        assert args.bai_num_clusters == 4
        assert args.bai_surrogate_model == 'rf'
        
        print("✓ Argument parsing tests passed")
    finally:
        sys.argv = original_argv


def main():
    """Run all tests."""
    print("Running BAI-FB integration tests...\n")
    
    try:
        test_arm_state()
        print()
        
        test_bai_methods()
        print()
        
        test_arm_builders()
        print()
        
        test_args_parsing()
        print()
        
        print("🎉 All tests passed! BAI-FB integration is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
