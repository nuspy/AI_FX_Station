"""
Neural network encoders for dimensionality reduction.

Provides VAE and Autoencoder implementations for feature compression.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Literal
from loguru import logger

# Import DeviceManager for GPU support
try:
    from ..utils.device_manager import DeviceManager, DeviceType
except ImportError:
    # Fallback if device_manager not available
    DeviceManager = None
    DeviceType = Literal["auto", "cuda", "cpu"]


class Autoencoder(nn.Module):
    """
    Simple autoencoder for dimensionality reduction.

    Architecture:
    - Encoder: input -> hidden layers -> latent (bottleneck)
    - Decoder: latent -> hidden layers -> output
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None):
        """
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space (compressed representation)
            hidden_dims: List of hidden layer dimensions (default: [128, 64])
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class VAE(nn.Module):
    """
    Variational Autoencoder for dimensionality reduction.

    Adds probabilistic encoding with KL divergence regularization.
    More robust than basic autoencoder for feature learning.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None):
        """
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions (default: [128, 64])
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        self.encoder_base = nn.Sequential(*encoder_layers)

        # Latent space parameters (mean and log variance)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Returns:
            recon: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class SklearnAutoencoder:
    """
    Scikit-learn compatible wrapper for PyTorch Autoencoder.
    Provides .fit(), .transform() interface like PCA.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        device: str = "auto",
        verbose: bool = True
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            device: Device ('auto', 'cuda', or 'cpu' - default: 'auto')
            verbose: Print training progress
        """
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Use DeviceManager for smart device selection
        if DeviceManager:
            self.device = DeviceManager.get_device(device)
            if self.verbose and self.device.type == "cuda":
                logger.info(f"🚀 Autoencoder will use GPU: {torch.cuda.get_device_name(0)}")
        else:
            # Fallback to legacy logic
            if device == "auto" or device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

        self.model = None
        self.input_dim_ = None
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def fit(self, X: np.ndarray, y=None) -> 'SklearnAutoencoder':
        """
        Fit autoencoder to data.

        Args:
            X: Input data (n_samples, n_features)
            y: Ignored (for sklearn compatibility)
        """
        X = np.asarray(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]

        # Normalize data
        self.scaler_mean_ = X.mean(axis=0)
        self.scaler_std_ = X.std(axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean_) / self.scaler_std_

        # Create model
        self.model = Autoencoder(
            input_dim=self.input_dim_,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Autoencoder Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to latent space.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Latent representations (n_samples, latent_dim)
        """
        X = np.asarray(X, dtype=np.float32)
        X_scaled = (X - self.scaler_mean_) / self.scaler_std_

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            z = self.model.encode(X_tensor)
            return z.cpu().numpy()

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode latent representations back to input space.

        Args:
            Z: Latent representations (n_samples, latent_dim)

        Returns:
            Reconstructed inputs (n_samples, input_dim)
        """
        self.model.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(Z).to(self.device)
            X_recon = self.model.decode(Z_tensor)
            X_recon_scaled = X_recon.cpu().numpy()
            return X_recon_scaled * self.scaler_std_ + self.scaler_mean_


class SklearnVAE:
    """
    Scikit-learn compatible wrapper for PyTorch VAE.
    Provides .fit(), .transform() interface like PCA.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        beta: float = 1.0,
        device: str = "auto",
        verbose: bool = True
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            beta: KL divergence weight (beta-VAE)
            device: Device ('auto', 'cuda', or 'cpu' - default: 'auto')
            verbose: Print training progress
        """
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.verbose = verbose

        # Use DeviceManager for smart device selection
        if DeviceManager:
            self.device = DeviceManager.get_device(device)
            if self.verbose and self.device.type == "cuda":
                logger.info(f"🚀 VAE will use GPU: {torch.cuda.get_device_name(0)}")
        else:
            # Fallback to legacy logic
            if device == "auto" or device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

        self.model = None
        self.input_dim_ = None
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def _vae_loss(self, recon_x, x, mu, logvar):
        """VAE loss = Reconstruction loss + KL divergence."""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld

    def fit(self, X: np.ndarray, y=None) -> 'SklearnVAE':
        """
        Fit VAE to data.

        Args:
            X: Input data (n_samples, n_features)
            y: Ignored (for sklearn compatibility)
        """
        X = np.asarray(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]

        # Normalize data
        self.scaler_mean_ = X.mean(axis=0)
        self.scaler_std_ = X.std(axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean_) / self.scaler_std_

        # Create model
        self.model = VAE(
            input_dim=self.input_dim_,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                optimizer.zero_grad()
                recon, mu, logvar = self.model(batch_x)
                loss = self._vae_loss(recon, batch_x, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(f"VAE Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to latent space (using mean of distribution).

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Latent representations (n_samples, latent_dim)
        """
        X = np.asarray(X, dtype=np.float32)
        X_scaled = (X - self.scaler_mean_) / self.scaler_std_

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            mu, _ = self.model.encode(X_tensor)
            return mu.cpu().numpy()

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode latent representations back to input space.

        Args:
            Z: Latent representations (n_samples, latent_dim)

        Returns:
            Reconstructed inputs (n_samples, input_dim)
        """
        self.model.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(Z).to(self.device)
            X_recon = self.model.decode(Z_tensor)
            X_recon_scaled = X_recon.cpu().numpy()
            return X_recon_scaled * self.scaler_std_ + self.scaler_mean_
