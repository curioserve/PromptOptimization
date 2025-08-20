from typing import List, Tuple, Literal, Optional
import numpy as np
from .datatypes import TrainingExample, ModelKind
from .expo_scorer import ExpoScorer
from .expo_es_scorer import ExpoEScorer
from .utils import DESC_INPUT_DIM, EMBED_DIM

class ScorerTrainer:
    """
    Keeps a growing history S_{t}, refits a scorer each iteration, and exposes batched prediction.
    """
    def __init__(self, kind: ModelKind, device: Optional[str] = None, lr: float = 1e-3):
        self.kind = kind
        self.history: List[TrainingExample] = []
        if kind == "EXPO":
            self.model = ExpoScorer(device=device, lr=lr)
            self.input_dim = DESC_INPUT_DIM
        elif kind == "EXPO_ES":
            self.model = ExpoEScorer(device=device, lr=lr)
            self.input_dim = EMBED_DIM
        else:
            raise ValueError("Unknown kind")

    def add_example(self, ex: TrainingExample):
        assert ex.x.shape[0] == self.input_dim, f"bad input dim: {ex.x.shape[0]} vs {self.input_dim}"
        self.history.append(ex)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.stack([ex.x for ex in self.history], axis=0).astype(np.float32)
        y = np.array([ex.y for ex in self.history], dtype=np.float32)
        return X, y

    def refit(self, epochs: int = 50, batch_size: int = 64, verbose: bool = False):
        X, y = self.as_arrays()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict_batch(self, X_candidates: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        return self.model.predict(X_candidates, batch_size=batch_size)

    def snapshot_params(self):
        """
        Save model parameters (for EXPO-ES ensemble across iterations, if you want that).
        """
        import copy
        return copy.deepcopy(self.model.state_dict())

    def load_params(self, state_dict):
        self.model.load_state_dict(state_dict)
