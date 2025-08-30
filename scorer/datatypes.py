from dataclasses import dataclass
import numpy as np
from typing import Literal, Optional

# What the scorer learns from each iteration
@dataclass
class TrainingExample:
    # For EXPO: x = concat(g(D), g(I)) with shape (6144,)
    # For EXPO-ES: x = g(exemplar) with shape (3072,)
    x: np.ndarray  # float32
    y: float       # scalar utility/score (higher = better)
    tag: Optional[str] = None  # e.g., "iter_12_D=.._I=.." or exemplar id

ModelKind = Literal["EXPO", "EXPO_ES"]
