
import numpy as np
import pytest

from scorer.trainer import ScorerTrainer
from scorer.datatypes import TrainingExample
from scorer.utils import set_deterministic, DESC_INPUT_DIM, EMBED_DIM

def _make_dataset(input_dim: int, n: int = 96, k_active: int = 16, noise: float = 0.01):
    """
    Synthetic data: y = sum(w * relu(x_active)) + noise.
    Keeps things light while still benefitting from a ReLU MLP.
    """
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, input_dim)).astype(np.float32)
    idx = rng.choice(input_dim, size=k_active, replace=False)
    w   = rng.normal(size=(k_active,)).astype(np.float32)

    X_active = np.maximum(X[:, idx], 0.0)  # ReLU on selected features
    y = X_active @ w + rng.normal(scale=noise, size=(n,)).astype(np.float32)
    y = y.astype(np.float32)
    return X, y

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

@pytest.mark.parametrize("kind,input_dim", [
    ("EXPO", DESC_INPUT_DIM),      # 6144
    ("EXPO_ES", EMBED_DIM),        # 3072
])
def test_predict_shapes_and_finiteness(kind, input_dim):
    set_deterministic(7)
    trainer = ScorerTrainer(kind=kind, device="cpu", lr=1e-3)

    # one tiny training example just to exercise the path
    X1 = np.zeros((1, input_dim), dtype=np.float32)
    y1 = np.array([0.0], dtype=np.float32)
    trainer.add_example(TrainingExample(x=X1[0], y=float(y1[0])))
    trainer.refit(epochs=2, batch_size=1, verbose=False)

    # batch prediction shape
    Xcand = np.random.default_rng(0).normal(size=(10, input_dim)).astype(np.float32)
    preds = trainer.predict_batch(Xcand)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))

@pytest.mark.parametrize("kind,input_dim,epochs,improve_ratio", [
    ("EXPO", DESC_INPUT_DIM, 20, 0.75),
    ("EXPO_ES", EMBED_DIM, 20, 0.70),
])
def test_training_improves_loss(kind, input_dim, epochs, improve_ratio):
    set_deterministic(42)
    X, y = _make_dataset(input_dim=input_dim, n=120, k_active=16, noise=0.02)

    trainer = ScorerTrainer(kind=kind, device="cpu", lr=1e-3)

    # add all examples to history
    for i in range(X.shape[0]):
        trainer.add_example(TrainingExample(x=X[i], y=float(y[i]), tag=f"ex_{i}"))

    # loss before training (random init)
    preds_before = trainer.predict_batch(X, batch_size=64)
    loss_before = _mse(preds_before, y)

    # train
    trainer.refit(epochs=epochs, batch_size=64, verbose=False)

    # loss after training
    preds_after = trainer.predict_batch(X, batch_size=64)
    loss_after = _mse(preds_after, y)

    assert loss_after < loss_before * improve_ratio, f"Loss did not improve enough: {loss_before} -> {loss_after}"

def test_snapshot_and_restore_expo_es():
    set_deterministic(99)
    kind = "EXPO_ES"
    input_dim = EMBED_DIM
    X, y = _make_dataset(input_dim=input_dim, n=80, k_active=12, noise=0.02)

    trainer = ScorerTrainer(kind=kind, device="cpu", lr=1e-3)

    # load a small history
    for i in range(X.shape[0]):
        trainer.add_example(TrainingExample(x=X[i], y=float(y[i])))

    # initial short fit
    trainer.refit(epochs=5, batch_size=32)
    preds_a = trainer.predict_batch(X, batch_size=64).copy()
    snap = trainer.snapshot_params()

    # further training changes predictions
    trainer.refit(epochs=10, batch_size=32)
    preds_b = trainer.predict_batch(X, batch_size=64).copy()

    # ensure they differ (very likely)
    diff = float(np.mean(np.abs(preds_a - preds_b)))
    assert diff > 1e-5

    # restore snapshot and verify predictions match snap-time preds
    trainer.load_params(snap)
    preds_c = trainer.predict_batch(X, batch_size=64).copy()
    # exact equality not guaranteed due to dtype/ops; use tight tolerance
    assert np.allclose(preds_a, preds_c, atol=1e-6), "Restored params did not reproduce saved predictions"
