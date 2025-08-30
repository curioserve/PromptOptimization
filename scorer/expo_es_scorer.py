from typing import Optional, Dict, Any
import numpy as np
import torch
from torch import nn
from .utils import EMBED_DIM, EXPO_ES_HIDDEN, init_mlp, to_tensor

class ExpoESMLP(nn.Module):
    """
    Input:  (batch, 3072)    # g(exemplar)
    Hidden: 512, ReLU
    Output: scalar (batch, 1)
    Loss:   MSE
    """
    def __init__(self, input_dim: int = EMBED_DIM, hidden: int = EXPO_ES_HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        init_mlp(self.fc1)
        init_mlp(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        y = self.fc2(h)
        return y.squeeze(-1)

class ExpoEScorer:
    """
    Same loop as EXPO but for exemplar embeddings. Can be snapshotted/ensembled across iterations.
    """
    def __init__(self, device: Optional[str] = None, lr: float = 1e-3, weight_decay: float = 0.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ExpoESMLP().to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 64, verbose: bool = False):
        self.model.train()
        ds = torch.utils.data.TensorDataset(to_tensor(X), to_tensor(y))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            total = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            if verbose:
                print(f"[EXPO-ES] epoch={epoch+1} loss={total/len(ds):.6f}")

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        out = np.empty((X.shape[0],), dtype=np.float32)
        for i in range(0, X.shape[0], batch_size):
            xb = to_tensor(X[i:i+batch_size]).to(self.device, non_blocking=True)
            out[i:i+batch_size] = self.model(xb).cpu().numpy()
        return out

    def state_dict(self) -> Dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, sd: Dict[str, Any]):
        self.model.load_state_dict(sd)
