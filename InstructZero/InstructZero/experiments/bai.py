"""Fixed-budget best-arm identification (BAI-FB) utilities for InstructZero.

This module provides lightweight data structures and algorithms to run
fixed-budget best-arm identification over a finite set of soft prompts.
The implementation focuses on sequential halving and continuous reject
(CR), which cover the low- to mid-sized candidate regimes we target in
InstructZero.  The functions are intentionally framework-agnostic: the
caller supplies callables that evaluate soft prompts and handle logging.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class ArmState:
    """Tracks statistics for a single candidate arm."""

    idx: int
    vector: Any
    key: Tuple[float, ...]
    pulls: int = 0
    rewards: List[float] = field(default_factory=list)
    instruction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self, reward: float, instruction: Optional[str] = None) -> None:
        self.pulls += 1
        self.rewards.append(float(reward))
        if instruction is not None:
            self.instruction = instruction

    @property
    def mean(self) -> float:
        if not self.rewards:
            return float("-inf")
        return sum(self.rewards) / len(self.rewards)

    def confidence_radius(self, total_pulls: int, delta: float) -> float:
        if self.pulls == 0:
            return float("inf")
        total = max(total_pulls, 2)
        return math.sqrt((2.0 * math.log(total / max(delta, 1e-12))) / self.pulls)

    def summary(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "pulls": self.pulls,
            "mean": self.mean,
            "instruction": self.instruction,
        }


def _log_arm_table(arms: Iterable[ArmState], logger: Callable[[str], None], prefix: str) -> None:
    for arm in arms:
        logger(
            f"{prefix}arm={arm.idx} pulls={arm.pulls} "
            f"mean={arm.mean:.4f} instruction={'None' if arm.instruction is None else arm.instruction[:60]}"
        )


def sequential_halving(
    arms: List[ArmState],
    evaluate_arm: Callable[[ArmState], Dict[str, Any]],
    total_budget: int,
    batch_size: int,
    logger: Callable[[str], None] = print,
) -> Tuple[ArmState, List[Dict[str, Any]]]:
    """Run Sequential Halving over ``arms`` under a fixed budget.

    Args:
        arms: Candidate arms to evaluate.
        evaluate_arm: Callable that evaluates an arm and updates its stats.
        total_budget: Number of available evaluations (after seeding).
        batch_size: Number of pulls per arm in each phase (minimum 1).
        logger: Logging callable (defaults to ``print``).

    Returns:
        The best arm according to the empirical mean and a history log.
    """
    history: List[Dict[str, Any]] = []
    if total_budget <= 0 or not arms:
        best = max(arms, key=lambda a: a.mean if a.pulls > 0 else float("-inf"))
        return best, history

    budget_remaining = total_budget
    active = list(arms)
    phase = 0

    while len(active) > 1 and budget_remaining > 0:
        pulls_per_arm = max(1, min(batch_size, budget_remaining // len(active) or 1))
        logger(f"[BAI][SH] phase={phase} active={len(active)} pulls_per_arm={pulls_per_arm} budget={budget_remaining}")
        for arm in active:
            for _ in range(pulls_per_arm):
                if budget_remaining <= 0:
                    break
                evaluate_arm(arm)
                budget_remaining -= 1
        _log_arm_table(active, logger, prefix="[BAI][SH] ")

        if len(active) == 1 or budget_remaining <= 0:
            break

        sorted_active = sorted(active, key=lambda a: a.mean, reverse=True)
        cutoff = max(1, len(sorted_active) // 2)
        survivors = sorted_active[:cutoff]
        eliminated = sorted_active[cutoff:]
        logger(
            "[BAI][SH] eliminating arms: "
            + ", ".join(f"{arm.idx}(mean={arm.mean:.3f})" for arm in eliminated)
        )
        active = survivors
        history.append(
            {
                "phase": phase,
                "active": [arm.idx for arm in active],
                "eliminated": [arm.idx for arm in eliminated],
                "snapshot": {arm.idx: arm.summary() for arm in arms},
            }
        )
        phase += 1

    best = max(active, key=lambda a: a.mean if a.pulls > 0 else float("-inf"))
    return best, history


def continuous_reject(
    arms: List[ArmState],
    evaluate_arm: Callable[[ArmState], Dict[str, Any]],
    total_budget: int,
    batch_size: int,
    delta: float,
    logger: Callable[[str], None] = print,
) -> Tuple[ArmState, List[Dict[str, Any]]]:
    """Run Continuous Reject (CR) over ``arms`` under a fixed budget."""
    history: List[Dict[str, Any]] = []
    if total_budget <= 0 or not arms:
        best = max(arms, key=lambda a: a.mean if a.pulls > 0 else float("-inf"))
        return best, history

    budget_remaining = total_budget
    active = list(arms)
    total_pulls = sum(arm.pulls for arm in arms)

    round_idx = 0
    while len(active) > 1 and budget_remaining > 0:
        logger(
            f"[BAI][CR] round={round_idx} active={len(active)} budget={budget_remaining}"
        )
        for arm in list(active):
            pulls = max(1, min(batch_size, budget_remaining))
            for _ in range(pulls):
                if budget_remaining <= 0:
                    break
                evaluate_arm(arm)
                budget_remaining -= 1
                total_pulls += 1
        _log_arm_table(active, logger, prefix="[BAI][CR] ")

        best_arm = max(active, key=lambda a: a.mean if a.pulls > 0 else float("-inf"))
        best_lower = best_arm.mean - best_arm.confidence_radius(total_pulls, delta)
        eliminated: List[ArmState] = []
        for arm in list(active):
            if arm is best_arm:
                continue
            upper = arm.mean + arm.confidence_radius(total_pulls, delta)
            if upper < best_lower:
                eliminated.append(arm)
                active.remove(arm)
        if eliminated:
            logger(
                "[BAI][CR] eliminating arms: "
                + ", ".join(f"{arm.idx}(mean={arm.mean:.3f})" for arm in eliminated)
            )
        history.append(
            {
                "round": round_idx,
                "active": [arm.idx for arm in active],
                "eliminated": [arm.idx for arm in eliminated],
                "snapshot": {arm.idx: arm.summary() for arm in arms},
            }
        )
        round_idx += 1

    best = max(active, key=lambda a: a.mean if a.pulls > 0 else float("-inf"))
    return best, history


def cluster_bai_in_cluster(
    arms: List[ArmState],
    evaluate_arm: Callable[[ArmState], Dict[str, Any]],
    total_budget: int,
    batch_size: int,
    num_clusters: int,
    embedding_model: Optional[str],
    logger: Callable[[str], None] = print,
) -> Tuple[ArmState, List[Dict[str, Any]]]:
    """Run Cluster → BAI → In-cluster BAI over ``arms`` under a fixed budget.
    
    This method first clusters arms based on instruction embeddings, then uses
    BAI to select the best cluster, and finally uses BAI within that cluster.
    """
    history: List[Dict[str, Any]] = []
    if total_budget <= 0 or not arms:
        best = max(arms, key=lambda a: a.mean if a.pulls > 0 else float("-inf"))
        return best, history
    
    # Phase 1: Cluster selection (30% of budget)
    cluster_budget = int(total_budget * 0.3)
    remaining_budget = total_budget - cluster_budget
    
    logger(f"[BAI][CLST] Phase 1: Cluster selection with budget {cluster_budget}")
    
    # For now, use random clustering if no embedding model provided
    if embedding_model is None:
        import random
        random.shuffle(arms)
        cluster_size = max(1, len(arms) // num_clusters)
        clusters = [arms[i:i + cluster_size] for i in range(0, len(arms), cluster_size)]
        if len(clusters) > num_clusters:
            # Merge last few clusters if we have too many
            clusters[-2] = clusters[-2] + clusters[-1]
            clusters = clusters[:-1]
    else:
        # TODO: Implement proper clustering with embeddings
        # For now, fall back to random clustering
        import random
        random.shuffle(arms)
        cluster_size = max(1, len(arms) // num_clusters)
        clusters = [arms[i:i + cluster_size] for i in range(0, len(arms), cluster_size)]
    
    # Evaluate each cluster with a few pulls
    cluster_means = {}
    for i, cluster in enumerate(clusters):
        if cluster_budget <= 0:
            break
        pulls_per_arm = max(1, min(batch_size, cluster_budget // len(cluster) // len(clusters)))
        cluster_sum = 0
        cluster_pulls = 0
        for arm in cluster:
            if cluster_budget <= 0:
                break
            for _ in range(pulls_per_arm):
                if cluster_budget <= 0:
                    break
                evaluate_arm(arm)
                cluster_sum += arm.mean
                cluster_pulls += 1
                cluster_budget -= 1
        cluster_means[i] = cluster_sum / max(cluster_pulls, 1)
        logger(f"[BAI][CLST] Cluster {i}: mean={cluster_means[i]:.4f}, arms={len(cluster)}")
    
    # Select best cluster
    best_cluster_idx = max(cluster_means.keys(), key=lambda k: cluster_means[k])
    best_cluster = clusters[best_cluster_idx]
    logger(f"[BAI][CLST] Selected cluster {best_cluster_idx} with mean {cluster_means[best_cluster_idx]:.4f}")
    
    # Phase 2: In-cluster BAI (70% of budget)
    logger(f"[BAI][CLST] Phase 2: In-cluster BAI with budget {remaining_budget}")
    best_arm, cluster_history = sequential_halving(
        best_cluster, evaluate_arm, remaining_budget, batch_size, logger
    )
    
    history.extend(cluster_history)
    return best_arm, history


def global_surrogate_elimination(
    arms: List[ArmState],
    evaluate_arm: Callable[[ArmState], Dict[str, Any]],
    total_budget: int,
    batch_size: int,
    surrogate_model: str,
    embedding_model: Optional[str],
    logger: Callable[[str], None] = print,
) -> Tuple[ArmState, List[Dict[str, Any]]]:
    """Run Global Surrogate Elimination over ``arms`` under a fixed budget.
    
    This method learns a surrogate model from instruction embeddings to scores,
    uses it to rank arms, and prunes aggressively before evaluating survivors.
    """
    history: List[Dict[str, Any]] = []
    if total_budget <= 0 or not arms:
        best = max(arms, key=lambda a: a.mean if a.pulls > 0 else float("-inf"))
        return best, history
    
    # Initial evaluation phase
    initial_budget = min(total_budget // 3, len(arms) * 2)
    logger(f"[BAI][GSE] Initial evaluation phase with budget {initial_budget}")
    
    for arm in arms[:initial_budget//batch_size]:
        for _ in range(batch_size):
            if initial_budget <= 0:
                break
            evaluate_arm(arm)
            initial_budget -= 1
    
    # Learn surrogate model
    if embedding_model is None:
        logger("[BAI][GSE] No embedding model provided, using random forest on arm indices")
        # Fallback: use arm indices as features
        X = np.array([[arm.idx] for arm in arms if arm.pulls > 0])
        y = np.array([arm.mean for arm in arms if arm.pulls > 0])
    else:
        # TODO: Implement proper embedding-based surrogate
        # For now, use arm indices
        X = np.array([[arm.idx] for arm in arms if arm.pulls > 0])
        y = np.array([arm.mean for arm in arms if arm.pulls > 0])
    
    if len(X) > 0:
        if surrogate_model == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
        else:  # linear
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        
        model.fit(X, y)
        
        # Predict scores for all arms
        X_all = np.array([[arm.idx] for arm in arms])
        predictions = model.predict(X_all)
        
        # Rank arms by predicted scores
        ranked_indices = np.argsort(predictions)[::-1]
        remaining_arms = [arms[i] for i in ranked_indices[:len(arms)//2]]  # Keep top half
        
        logger(f"[BAI][GSE] Surrogate model trained, selected {len(remaining_arms)} arms for final evaluation")
    else:
        remaining_arms = arms
    
    # Final evaluation with remaining budget
    remaining_budget = total_budget - (total_budget // 3)
    if remaining_budget > 0:
        best_arm, final_history = sequential_halving(
            remaining_arms, evaluate_arm, remaining_budget, batch_size, logger
        )
        history.extend(final_history)
    else:
        best_arm = max(arms, key=lambda a: a.mean if a.pulls > 0 else float("-inf"))
    
    return best_arm, history


__all__ = ["ArmState", "sequential_halving", "continuous_reject", "cluster_bai_in_cluster", "global_surrogate_elimination"]