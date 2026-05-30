"""
models.py
=========
PyTorch implementation of the 4-layer bidirectional LSTM model from Daly (2023).

Architecture
------------
- Input:  124 features per time step  (4 sources × 31 EEG channels)
- 4 × BiLSTM layers, 250 hidden units each  → 500 units per step (both directions)
- Fully-connected output layer (regression) → 1 output per time step
- Loss: MSE

Input tensor shape: (batch, seq_len, 124)
Target tensor shape: (batch, seq_len, 1)  — audio waveform at 100 Hz
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiLSTMDecoder(nn.Module):
    """4-layer bidirectional LSTM for music reconstruction from EEG features.

    Parameters
    ----------
    input_size : int
        Number of input features per time step (124 in the paper).
    hidden_size : int
        Number of hidden units in each LSTM layer (250 in the paper).
    num_layers : int
        Number of stacked LSTM layers (4 in the paper).
    output_size : int
        Dimensionality of the output (1 for mono audio).
    dropout : float
        Dropout probability applied between LSTM layers (0 = disabled).
    """

    def __init__(
        self,
        input_size: int = 124,
        hidden_size: int = 250,
        num_layers: int = 4,
        output_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiLSTM output is 2 × hidden_size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)
        hx : tuple of tensors or None
            Initial hidden and cell states.

        Returns
        -------
        out : torch.Tensor, shape (batch, seq_len, output_size)
        (h_n, c_n) : tuple
            Final hidden and cell states.
        """
        lstm_out, (h_n, c_n) = self.lstm(x, hx)  # (batch, seq_len, 2*hidden)
        out = self.fc(lstm_out)                    # (batch, seq_len, output_size)
        return out, (h_n, c_n)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: forward pass without returning hidden state."""
        out, _ = self.forward(x)
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: BiLSTMDecoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    clip_grad_norm: float = 1.0,
    scheduler_patience: int = 10,
    verbose: bool = True,
) -> Tuple[BiLSTMDecoder, dict]:
    """Train the biLSTM model.

    Parameters
    ----------
    model : BiLSTMDecoder
    X_train : np.ndarray, shape (n_trials, seq_len, 124)
        Feature matrices (already transposed to seq_len-first).
    y_train : np.ndarray, shape (n_trials, seq_len) or (n_trials, seq_len, 1)
        Target audio waveforms at 100 Hz.
    n_epochs : int
    batch_size : int
    lr : float
        Initial learning rate.
    weight_decay : float
        L2 regularisation.
    device : str or None
        ``'cuda'``, ``'cpu'``, or None (auto-detect).
    clip_grad_norm : float
        Gradient clipping threshold.
    scheduler_patience : int
        Epochs without improvement before LR is halved.
    verbose : bool

    Returns
    -------
    model : BiLSTMDecoder
        Trained model (in eval mode).
    history : dict
        ``{'train_loss': [...]}``
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Prepare tensors
    X_t = torch.FloatTensor(X_train)
    if y_train.ndim == 2:
        y_t = torch.FloatTensor(y_train).unsqueeze(-1)   # (n, T, 1)
    else:
        y_t = torch.FloatTensor(y_train)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=0.5, verbose=verbose
    )

    history = {"train_loss": []}

    model.train()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred, _ = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()

            if clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(dataset)
        history["train_loss"].append(epoch_loss)
        scheduler.step(epoch_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:4d}/{n_epochs}  loss={epoch_loss:.6f}"
                  f"  lr={optimizer.param_groups[0]['lr']:.2e}")

    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model: BiLSTMDecoder,
    X: np.ndarray,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Run inference with a trained model.

    Parameters
    ----------
    model : BiLSTMDecoder
    X : np.ndarray, shape (n_trials, seq_len, 124)
    batch_size : int
    device : str or None

    Returns
    -------
    predictions : np.ndarray, shape (n_trials, seq_len)
        Reconstructed audio envelopes.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    X_t = torch.FloatTensor(X)
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            out, _ = model(X_batch)         # (batch, seq_len, 1)
            preds.append(out.squeeze(-1).cpu().numpy())

    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def save_model(model: BiLSTMDecoder, path: str) -> None:
    """Save model weights and hyperparameters."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": {
                "input_size": model.input_size,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "output_size": model.output_size,
            },
        },
        path,
    )


def load_model(path: str) -> BiLSTMDecoder:
    """Load model from saved checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model = BiLSTMDecoder(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
