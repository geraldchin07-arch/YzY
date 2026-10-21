"""Small composition-token Transformer model for CrystalCast.

This is a numeric TransformerEncoder, not a language model: each alloy element is
a token, its scaled concentration is the token value, and a learned embedding
identifies which element the token represents.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class _CompositionEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int = 16, nhead: int = 2, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: batch x element-features -> batch x tokens x embedding-dimension.
        tokens = self.value_projection(x.unsqueeze(-1)) + self.feature_embedding
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


class CompositionTransformerRegressor:
    """Small CPU-friendly Transformer regressor with a scikit-learn-like API.

    Early stopping uses a validation slice of the training data. The held-out
    test set remains untouched by evaluate_model.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_epochs: int = 100,
        patience: int = 14,
        batch_size: int = 32,
        learning_rate: float = 0.003,
        weight_decay: float = 0.02,
    ):
        self.random_state = int(random_state)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)

    @staticmethod
    def _frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        return frame.replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y):
        frame = self._frame(X)
        self.feature_names_ = list(map(str, frame.columns))
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        values = self.imputer_.fit_transform(frame).astype(np.float32)
        self.x_scaler_ = StandardScaler()
        values = self.x_scaler_.fit_transform(values).astype(np.float32)

        target = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(values) != len(target):
            raise ValueError("X and y must contain the same number of rows.")
        if len(values) < 10:
            raise ValueError("The Transformer requires at least 10 labeled rows.")

        self.y_mean_ = float(np.mean(target))
        self.y_scale_ = float(np.std(target)) or 1.0
        scaled_y = ((target - self.y_mean_) / self.y_scale_).astype(np.float32)

        train_idx, val_idx = train_test_split(
            np.arange(len(values)), test_size=0.18, random_state=self.random_state
        )

        torch.manual_seed(self.random_state)
        try:
            torch.set_num_threads(min(torch.get_num_threads(), 4))
        except RuntimeError:
            # Torch can reject thread-count changes if parallel work has started.
            pass

        self.model_ = _CompositionEncoder(values.shape[1])
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        x_train = torch.tensor(values[train_idx], dtype=torch.float32)
        y_train = torch.tensor(scaled_y[train_idx], dtype=torch.float32)
        x_val = torch.tensor(values[val_idx], dtype=torch.float32)
        y_val = torch.tensor(scaled_y[val_idx], dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=min(self.batch_size, len(x_train)),
            shuffle=True,
            generator=generator,
        )

        best_loss = float("inf")
        best_state = copy.deepcopy(self.model_.state_dict())
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model_(batch_x)
                loss = nn.functional.mse_loss(prediction, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_prediction = self.model_(x_val)
                val_loss = float(nn.functional.mse_loss(val_prediction, y_val).item())

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.n_epochs_trained_ = epoch + 1
        self.best_epoch_ = best_epoch
        return self

    def predict(self, X):
        frame = self._frame(X)
        frame.columns = list(map(str, frame.columns))
        if list(frame.columns) != self.feature_names_:
            missing = sorted(set(self.feature_names_) - set(frame.columns))
            extra = sorted(set(frame.columns) - set(self.feature_names_))
            if missing or extra:
                raise ValueError(
                    f"Composition feature columns changed (missing={missing}, extra={extra})."
                )
            frame = frame[self.feature_names_]

        values = self.imputer_.transform(frame).astype(np.float32)
        values = self.x_scaler_.transform(values).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(torch.tensor(values, dtype=torch.float32)).cpu().numpy()
        return predictions * self.y_scale_ + self.y_mean_
