import numpy as np
import torch

EMBED_DIM = 3072
DESC_INPUT_DIM = EMBED_DIM * 2           # 6144 for D ⊕ I
EXPO_HIDDEN = 1536                       # 1 hidden layer (EXPO)
EXPO_ES_HIDDEN = 512                     # 1 hidden layer (EXPO-ES)

def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))

def set_deterministic(seed: int = 13):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def init_mlp(layer: torch.nn.Linear):
    # Kaiming uniform works well with ReLU
    torch.nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
    torch.nn.init.zeros_(layer.bias)
