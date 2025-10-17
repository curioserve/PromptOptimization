"""
Soft Prompt Pool System for InstructZero

This module implements a comprehensive soft-prompt pool system with:
1. 10k soft-prompt pool builder with Latin Hypercube sampling
2. Lazy evaluation entrypoint with caching
3. Screening step to shortlist 10k → K_active

The system is designed to efficiently manage large pools of soft prompts
while minimizing expensive evaluation calls through smart caching and screening.
"""

import os
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Arm:
    """Represents a single soft prompt arm with latent vector p."""
    
    idx: int
    p: np.ndarray  # Latent vector p ∈ R^d
    key: Tuple[float, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate a unique key from the vector for caching."""
        if not self.key:
            self.key = tuple(self.p.flatten().tolist())
    
    def __hash__(self):
        """Make Arm hashable for caching."""
        return hash(self.key)
    
    def __eq__(self, other):
        """Equality based on the vector key."""
        if not isinstance(other, Arm):
            return False
        return self.key == other.key


class SoftPromptPool:
    """Manages a pool of soft prompt arms with efficient storage and retrieval."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the soft prompt pool.
        
        Args:
            cache_dir: Directory for caching pools and evaluations. If None, uses in-memory only.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self._pool_cache: Dict[str, List[Arm]] = {}
        self._evaluation_cache: Dict[Tuple[float, ...], Tuple[float, str]] = {}
        
        # Evaluation function (set by user)
        self._evaluate_fn: Optional[Callable[[np.ndarray], Tuple[float, str]]] = None
    
    def build_pool(
        self, 
        K_init: int = 10000, 
        d: int = 10, 
        sampler: str = "lhs",
        seed: int = 42
    ) -> List[Arm]:
        """
        Build a pool of K_init soft prompt arms.
        
        Args:
            K_init: Number of arms to generate (default: 10000)
            d: Dimension of latent vectors (default: 10)
            sampler: Sampling method - "lhs" (Latin Hypercube) or "uniform" (default: "lhs")
            seed: Random seed for reproducibility (default: 42)
            
        Returns:
            List of Arm objects with latent vectors p ∈ R^d
        """
        cache_key = f"pool_K{K_init}_d{d}_{sampler}_seed{seed}"
        
        # Check cache first
        if cache_key in self._pool_cache:
            logger.info(f"Loading pool from memory cache: {cache_key}")
            return self._pool_cache[cache_key]
        
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                logger.info(f"Loading pool from disk cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    arms = pickle.load(f)
                self._pool_cache[cache_key] = arms
                return arms
        
        logger.info(f"Generating new pool: K={K_init}, d={d}, sampler={sampler}, seed={seed}")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Generate vectors based on sampler
        if sampler == "lhs":
            # Latin Hypercube Sampling
            from scipy.stats import qmc
            sampler_obj = qmc.LatinHypercube(d=d, seed=seed)
            samples = sampler_obj.random(n=K_init)
            # Scale to [-1, 1] range
            p_vectors = 2 * samples - 1
        elif sampler == "uniform":
            # Uniform random sampling
            p_vectors = np.random.uniform(-1, 1, size=(K_init, d))
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Choose 'lhs' or 'uniform'.")
        
        # Create Arm objects
        arms = []
        for i, p in enumerate(p_vectors):
            arm = Arm(idx=i, p=p.astype(np.float32))
            arms.append(arm)
        
        # Cache the result
        self._pool_cache[cache_key] = arms
        
        if self.cache_dir:
            logger.info(f"Saving pool to disk cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(arms, f)
        
        logger.info(f"Generated pool of {len(arms)} arms")
        return arms
    
    def set_evaluation_function(self, evaluate_fn: Callable[[np.ndarray], Tuple[float, str]]):
        """
        Set the evaluation function for lazy evaluation.
        
        Args:
            evaluate_fn: Function that takes a vector p and returns (score, instruction_text)
        """
        self._evaluate_fn = evaluate_fn
        logger.info("Evaluation function set")
    
    def evaluate_p(self, p: np.ndarray, force_regenerate: bool = False) -> Tuple[float, str]:
        """
        Lazy evaluation of a soft prompt vector p.
        
        On first call: generates instruction with g, queries f on mini-batch, caches result.
        On later calls: reuses cached instruction, recomputes score on new mini-batch.
        
        Args:
            p: Latent vector p ∈ R^d
            force_regenerate: If True, regenerate instruction even if cached
            
        Returns:
            Tuple of (score, instruction_text)
        """
        if self._evaluate_fn is None:
            raise ValueError("Evaluation function not set. Call set_evaluation_function() first.")
        
        # Create a key for caching
        p_key = tuple(p.flatten().tolist())
        
        # Check cache
        if not force_regenerate and p_key in self._evaluation_cache:
            score, instruction = self._evaluation_cache[p_key]
            logger.debug(f"Using cached evaluation for vector {p_key[:5]}...")
            return score, instruction
        
        # Evaluate the vector
        logger.debug(f"Evaluating new vector {p_key[:5]}...")
        score, instruction = self._evaluate_fn(p)
        
        # Cache the result
        self._evaluation_cache[p_key] = (score, instruction)
        
        # Optionally save to disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / "evaluations.pkl"
            try:
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        disk_cache = pickle.load(f)
                else:
                    disk_cache = {}
                
                disk_cache[p_key] = (score, instruction)
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(disk_cache, f)
            except Exception as e:
                logger.warning(f"Failed to save evaluation to disk cache: {e}")
        
        return score, instruction
    
    def screen_pool(
        self, 
        pool: List[Arm], 
        K_active: int = 1000,
        M_warmup: int = 512,
        surrogate_type: str = "mlp",
        seed: int = 42
    ) -> List[Arm]:
        """
        Screen the pool to select K_active arms using surrogate modeling.
        
        This implements the "GSE pre-screen" idea:
        1. Warm-up: evaluate M random arms
        2. Fit surrogate model on (p, score) pairs
        3. Rank all arms by surrogate prediction
        4. Select top K_active arms
        
        Args:
            pool: List of all arms to screen
            K_active: Number of arms to select for active set
            M_warmup: Number of random arms to evaluate for warm-up
            surrogate_type: Type of surrogate model - "mlp", "linear", or "rf"
            seed: Random seed for reproducibility
            
        Returns:
            List of top K_active arms selected by surrogate model
        """
        logger.info(f"Screening pool of {len(pool)} arms to select {K_active} active arms")
        
        if len(pool) <= K_active:
            logger.info("Pool size <= K_active, returning all arms")
            return pool
        
        # Set random seed
        np.random.seed(seed)
        
        # Step 1: Warm-up evaluation
        logger.info(f"Warm-up evaluation: sampling {M_warmup} random arms")
        warmup_indices = np.random.choice(len(pool), size=min(M_warmup, len(pool)), replace=False)
        warmup_arms = [pool[i] for i in warmup_indices]
        
        # Evaluate warm-up arms
        X_warmup = []
        y_warmup = []
        for arm in warmup_arms:
            score, _ = self.evaluate_p(arm.p)
            X_warmup.append(arm.p)
            y_warmup.append(score)
        
        X_warmup = np.array(X_warmup)
        y_warmup = np.array(y_warmup)
        
        logger.info(f"Warm-up completed. Score range: [{y_warmup.min():.4f}, {y_warmup.max():.4f}]")
        
        # Step 2: Fit surrogate model
        logger.info(f"Fitting {surrogate_type} surrogate model")
        surrogate = self._fit_surrogate(X_warmup, y_warmup, surrogate_type)
        
        # Step 3: Predict scores for all arms
        logger.info("Predicting scores for all arms")
        X_all = np.array([arm.p for arm in pool])
        predictions = surrogate.predict(X_all)
        
        # Step 4: Select top K_active arms
        top_indices = np.argsort(predictions)[::-1][:K_active]
        active_arms = [pool[i] for i in top_indices]
        
        logger.info(f"Selected {len(active_arms)} active arms. "
                   f"Predicted score range: [{predictions[top_indices].min():.4f}, {predictions[top_indices].max():.4f}]")
        
        return active_arms
    
    def _fit_surrogate(self, X: np.ndarray, y: np.ndarray, surrogate_type: str):
        """Fit a surrogate model on the warm-up data."""
        if surrogate_type == "linear":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif surrogate_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        elif surrogate_type == "mlp":
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")
        
        model.fit(X, y)
        return model
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the current cache state."""
        return {
            "pool_cache_size": len(self._pool_cache),
            "evaluation_cache_size": len(self._evaluation_cache),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None
        }
    
    def clear_cache(self, clear_disk: bool = False):
        """Clear the in-memory caches."""
        self._pool_cache.clear()
        self._evaluation_cache.clear()
        logger.info("In-memory cache cleared")
        
        if clear_disk and self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Disk cache cleared")


# Convenience functions for easy usage
def build_pool(K_init: int = 10000, d: int = 10, sampler: str = "lhs", seed: int = 42) -> List[Arm]:
    """
    Build a pool of K_init soft prompt arms.
    
    Args:
        K_init: Number of arms to generate (default: 10000)
        d: Dimension of latent vectors (default: 10)
        sampler: Sampling method - "lhs" (Latin Hypercube) or "uniform" (default: "lhs")
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        List of Arm objects with latent vectors p ∈ R^d
    """
    pool_manager = SoftPromptPool()
    return pool_manager.build_pool(K_init, d, sampler, seed)


def screen_pool(
    pool: List[Arm], 
    evaluate_fn: Callable[[np.ndarray], Tuple[float, str]],
    K_active: int = 1000,
    M_warmup: int = 512,
    surrogate_type: str = "mlp",
    seed: int = 42
) -> List[Arm]:
    """
    Screen a pool of arms to select the top K_active using surrogate modeling.
    
    Args:
        pool: List of all arms to screen
        evaluate_fn: Function that evaluates a vector p and returns (score, instruction)
        K_active: Number of arms to select for active set
        M_warmup: Number of random arms to evaluate for warm-up
        surrogate_type: Type of surrogate model - "mlp", "linear", or "rf"
        seed: Random seed for reproducibility
        
    Returns:
        List of top K_active arms selected by surrogate model
    """
    pool_manager = SoftPromptPool()
    pool_manager.set_evaluation_function(evaluate_fn)
    return pool_manager.screen_pool(pool, K_active, M_warmup, surrogate_type, seed)


# Example usage and testing
if __name__ == "__main__":
    # Example evaluation function (replace with your actual implementation)
    def dummy_evaluate_fn(p: np.ndarray) -> Tuple[float, str]:
        """Dummy evaluation function for testing."""
        # Simulate some computation
        score = np.sum(p**2) + np.random.normal(0, 0.1)
        instruction = f"Generated instruction for vector with norm {np.linalg.norm(p):.3f}"
        return float(score), instruction
    
    # Test the system
    print("Testing Soft Prompt Pool System")
    print("=" * 50)
    
    # 1. Build a small pool for testing
    print("1. Building pool...")
    pool = build_pool(K_init=100, d=5, sampler="lhs", seed=42)
    print(f"   Generated {len(pool)} arms")
    print(f"   First arm vector shape: {pool[0].p.shape}")
    print(f"   First arm vector: {pool[0].p}")
    
    # 2. Test lazy evaluation
    print("\n2. Testing lazy evaluation...")
    pool_manager = SoftPromptPool()
    pool_manager.set_evaluation_function(dummy_evaluate_fn)
    
    # Evaluate first arm
    score1, instruction1 = pool_manager.evaluate_p(pool[0].p)
    print(f"   First evaluation: score={score1:.4f}, instruction='{instruction1[:50]}...'")
    
    # Evaluate same arm again (should use cache)
    score2, instruction2 = pool_manager.evaluate_p(pool[0].p)
    print(f"   Cached evaluation: score={score2:.4f}, instruction='{instruction2[:50]}...'")
    print(f"   Cache hit: {score1 == score2}")
    
    # 3. Test screening
    print("\n3. Testing screening...")
    active_arms = pool_manager.screen_pool(pool, K_active=10, M_warmup=20, surrogate_type="linear")
    print(f"   Selected {len(active_arms)} active arms from {len(pool)} total")
    
    # 4. Cache statistics
    print("\n4. Cache statistics:")
    stats = pool_manager.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nTest completed successfully!")
