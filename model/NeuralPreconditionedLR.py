"""Hybrid neural-preconditioned linear regression via FGMRES."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn
from performer_pytorch import FastAttention
from torch.utils.data import DataLoader, TensorDataset


class _GatedGuideNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_attention: bool = False,
        attention_heads: int = 4,
        attention_dim: int = 32,
        attention_backend: str = "performer",
    ):
        super().__init__()
        self.use_attention = use_attention
        self.attention_backend = attention_backend
        self.gate_linear = nn.Linear(input_dim, input_dim)

        if self.use_attention:
            if attention_dim % attention_heads != 0:
                raise ValueError("attention_dim must be divisible by attention_heads")
            self.attn_in = nn.Linear(1, attention_dim)
            self.attn_out = nn.Linear(attention_dim, 1)
            self.attn_heads = attention_heads
            self.attn_dim = attention_dim
            if self.attention_backend == "performer":
                head_dim = attention_dim // attention_heads
                self.attn_qkv = nn.Linear(attention_dim, attention_dim * 3, bias=False)
                self.attn = FastAttention(dim_heads=head_dim, nb_features=None, causal=False)
            elif self.attention_backend == "mhsa":
                self.attn = nn.MultiheadAttention(
                    embed_dim=attention_dim,
                    num_heads=attention_heads,
                    batch_first=True,
                )
            else:
                raise ValueError("attention_backend must be 'performer' or 'mhsa'")

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.predictor = nn.Sequential(*layers)

    def _attention_logits(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(-1)
        attn_in = self.attn_in(x_seq)
        if self.attention_backend == "performer":
            bsz, seq_len, _ = attn_in.shape
            qkv = self.attn_qkv(attn_in)
            q, k, v = qkv.chunk(3, dim=-1)
            head_dim = self.attn_dim // self.attn_heads
            q = q.view(bsz, seq_len, self.attn_heads, head_dim).permute(0, 2, 1, 3)
            k = k.view(bsz, seq_len, self.attn_heads, head_dim).permute(0, 2, 1, 3)
            v = v.view(bsz, seq_len, self.attn_heads, head_dim).permute(0, 2, 1, 3)
            attn_out = self.attn(q, k, v)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.attn_dim)
        else:
            attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        return self.attn_out(attn_out).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_linear(x)
        if self.use_attention:
            gate_logits = gate_logits + self._attention_logits(x)
        gate = torch.sigmoid(gate_logits)
        gated = x * gate
        return self.predictor(gated)

    def gate_vector(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_linear(x)
        if self.use_attention:
            gate_logits = gate_logits + self._attention_logits(x)
        return torch.sigmoid(gate_logits)


@dataclass
class NeuralPreconditionedLinearRegression:
    """Hybrid solver: NN-guided preconditioner + linear FGMRES solve."""

    input_dim: int
    guide_hidden_dim: int = 64
    guide_dropout: float = 0.0
    guide_use_attention: bool = False
    guide_attention_heads: int = 4
    guide_attention_dim: int = 32
    guide_attention_backend: str = "performer"
    preconditioner: str = "nn"  # "nn", "jacobi", "none"
    device: str = "cpu"
    l2: float = 0.0
    random_state: int = 0

    def __post_init__(self) -> None:
        self._device = torch.device(self.device)
        self.w = np.zeros(self.input_dim, dtype=np.float64)
        self.b = 0.0
        self.guide = _GatedGuideNet(
            input_dim=self.input_dim,
            hidden_dim=self.guide_hidden_dim,
            dropout=self.guide_dropout,
            use_attention=self.guide_use_attention,
            attention_heads=self.guide_attention_heads,
            attention_dim=self.guide_attention_dim,
            attention_backend=self.guide_attention_backend,
        ).to(self._device)
        self._fitted = False
        self._gate_cache: torch.Tensor | None = None
        self.metrics_: dict[str, list[float] | float] = {}
        self._jacobi_inv: np.ndarray | None = None

    def _to_tensor(self, X: np.ndarray, y: np.ndarray | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self._device)
        y_t = None
        if y is not None:
            y_t = torch.as_tensor(y, dtype=torch.float32, device=self._device).view(-1, 1)
        return X_t, y_t

    def _train_guide(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> dict[str, list[float]]:
        print("[Guide] Starting training")
        self.guide.train()
        optimizer = torch.optim.Adam(self.guide.parameters(), lr=lr)
        criterion = nn.MSELoss()
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        loss_history: list[float] = []
        r2_history: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            preds_all: list[torch.Tensor] = []
            targets_all: list[torch.Tensor] = []
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.guide(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1
                preds_all.append(preds.detach().cpu())
                targets_all.append(y_batch.detach().cpu())
            if batch_count > 0:
                loss_history.append(epoch_loss / batch_count)
                y_true = torch.cat(targets_all, dim=0).numpy().reshape(-1)
                y_pred = torch.cat(preds_all, dim=0).numpy().reshape(-1)
                r2_history.append(self._r2_score(y_true, y_pred))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                latest_loss = loss_history[-1] if loss_history else float("nan")
                latest_r2 = r2_history[-1] if r2_history else float("nan")
                print(f"[Guide] Epoch {epoch + 1}/{epochs} | loss={latest_loss:.6f} | r2={latest_r2:.4f}")
        return {"loss": loss_history, "r2": r2_history}

    def _gate_preconditioner(self, r: np.ndarray) -> np.ndarray:
        """Apply the frozen gate to a residual vector r (shape: [p])."""
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self._device).view(1, -1)
        with torch.no_grad():
            gate = self.guide.gate_vector(r_t).view(-1)
        self._gate_cache = gate.detach().cpu()
        return (gate.detach().cpu().numpy() * r)

    def _jacobi_preconditioner(self, r: np.ndarray) -> np.ndarray:
        if self._jacobi_inv is None:
            return r
        return self._jacobi_inv * r

    def _select_preconditioner(self) -> Callable[[np.ndarray], np.ndarray]:
        if self.preconditioner == "nn":
            return self._gate_preconditioner
        if self.preconditioner == "jacobi":
            return self._jacobi_preconditioner
        if self.preconditioner == "none":
            return lambda r: r
        raise ValueError("preconditioner must be 'nn', 'jacobi', or 'none'")

    @staticmethod
    def _train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float,
        random_state: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        n = X.shape[0]
        idx = rng.permutation(n)
        split = int(n * (1 - test_size))
        train_idx, test_idx = idx[:split], idx[split:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    @staticmethod
    def _fgmres(
        A: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
        M: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        max_iter: int,
        tol: float,
        restart: int,
    ) -> Tuple[np.ndarray, list[float]]:
        x = x0.copy()
        history: list[float] = []

        def _norm(v: np.ndarray) -> float:
            return float(np.linalg.norm(v))

        iters_done = 0
        while iters_done < max_iter:
            r = b - A(x)
            beta = _norm(r)
            if beta < tol:
                history.append(beta)
                return x, history

            V: list[np.ndarray] = [r / beta]
            Z: list[np.ndarray] = []
            H = np.zeros((restart + 1, restart), dtype=np.float64)
            g = np.zeros(restart + 1, dtype=np.float64)
            g[0] = beta

            for j in range(restart):
                z = M(V[j])
                Z.append(z)
                w = A(z)
                for i in range(j + 1):
                    H[i, j] = np.dot(w, V[i])
                    w = w - H[i, j] * V[i]
                H[j + 1, j] = _norm(w)
                if H[j + 1, j] != 0 and j + 1 < restart:
                    V.append(w / H[j + 1, j])

                y, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], g[: j + 2], rcond=None)
                x_candidate = x + sum(y[i] * Z[i] for i in range(j + 1))
                res_norm = _norm(b - A(x_candidate))
                history.append(res_norm)
                iters_done += 1
                if res_norm < tol or iters_done >= max_iter:
                    return x_candidate, history

            y, *_ = np.linalg.lstsq(H[: restart + 1, : restart], g[: restart + 1], rcond=None)
            x = x + sum(y[i] * Z[i] for i in range(restart))

        return x, history

    @staticmethod
    def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(np.float64, copy=False)
        y_pred = y_pred.astype(np.float64, copy=False)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        guide_epochs: int = 10,
        guide_lr: float = 1e-3,
        guide_batch_size: int = 256,
        max_iter: int = 50,
        restart: int = 10,
        tol: float = 1e-6,
    ) -> "NeuralPreconditionedLinearRegression":
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X.shape[1] != self.input_dim:
            raise ValueError("X has different feature dimension than input_dim.")
        X_train, _, y_train, _ = self._train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        X_t, y_t = self._to_tensor(X_train, y_train)
        if y_t is None:
            raise ValueError("y must be provided for training.")

        # 1) Train the guide network on the training set.
        guide_metrics = self._train_guide(
            X_t,
            y_t,
            epochs=guide_epochs,
            batch_size=guide_batch_size,
            lr=guide_lr,
        )

        # 2) Freeze guide and build preconditioner function.
        self.guide.eval()

        # 3) Define linear system A w = b (normal equations).
        X_train_f64 = X_train.astype(np.float64, copy=False)
        y_train_f64 = y_train.astype(np.float64, copy=False)

        def A(v: np.ndarray) -> np.ndarray:
            Av = X_train_f64.T @ (X_train_f64 @ v)
            if self.l2 > 0:
                Av = Av + self.l2 * v
            return Av

        b_vec = X_train_f64.T @ y_train_f64

        # Jacobi preconditioner (diag of A)
        if self.preconditioner == "jacobi":
            diag = np.sum(X_train_f64 ** 2, axis=0)
            if self.l2 > 0:
                diag = diag + self.l2
            self._jacobi_inv = 1.0 / (diag + 1e-8)

        # 4) Solve with FGMRES using the NN gate as preconditioner.
        x0 = np.zeros(self.input_dim, dtype=np.float64)
        w_hat, fgmres_history = self._fgmres(
            A=A,
            b=b_vec,
            M=self._select_preconditioner(),
            x0=x0,
            max_iter=max_iter,
            tol=tol,
            restart=restart,
        )

        self.w = w_hat
        self.b = 0.0
        self._fitted = True
        train_preds = X_train_f64 @ self.w + self.b
        self.metrics_ = {
            "guide_train_loss": guide_metrics["loss"],
            "guide_train_r2": guide_metrics["r2"],
            "fgmres_residuals": fgmres_history,
            "train_r2": self._r2_score(y_train_f64, train_preds),
            "preconditioner": self.preconditioner,
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return (X @ self.w + self.b).reshape(-1)
