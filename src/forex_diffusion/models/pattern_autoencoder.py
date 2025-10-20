"""
ML-based pattern detection using Variational Autoencoder.

Detects anomalies and recurring patterns in price data through
unsupervised learning of latent representations.
"""
from __future__ import annotations

from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class PatternEncoder(nn.Module):
    """
    Encoder network for pattern detection.

    Compresses price sequences into low-dimensional latent space.
    """

    def __init__(
        self,
        input_channels: int = 7,
        sequence_length: int = 64,
        latent_dim: int = 32,
        hidden_dims: List[int] = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim

        # Convolutional encoder
        layers = []
        in_channels = input_channels

        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_channels = h_dim

        self.encoder = nn.Sequential(*layers)

        # Calculate flattened size
        self.flat_size = self._get_flat_size()

        # Latent projections (VAE)
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

    def _get_flat_size(self) -> int:
        """Calculate flattened size after convolutions."""
        x = torch.randn(1, self.input_channels, self.sequence_length)
        x = self.encoder(x)
        return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution.

        Args:
            x: Input tensor (B, C, L)

        Returns:
            (mu, logvar) for latent distribution
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class PatternDecoder(nn.Module):
    """
    Decoder network for pattern reconstruction.

    Reconstructs price sequences from latent representations.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        output_channels: int = 7,
        sequence_length: int = 64,
        hidden_dims: List[int] = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.sequence_length = sequence_length

        # Calculate initial size after upsampling
        self.init_length = sequence_length // (2 ** len(hidden_dims))
        self.init_channels = hidden_dims[0]

        # Projection from latent
        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_length)

        # Transposed convolutions
        layers = []
        in_channels = hidden_dims[0]

        for h_dim in hidden_dims[1:]:
            layers.extend([
                nn.ConvTranspose1d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_channels = h_dim

        # Final layer
        layers.append(nn.ConvTranspose1d(in_channels, output_channels, kernel_size=4, stride=2, padding=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstruction.

        Args:
            z: Latent tensor (B, latent_dim)

        Returns:
            Reconstructed sequence (B, C, L)
        """
        h = self.fc(z)
        h = h.view(h.size(0), self.init_channels, self.init_length)
        x_rec = self.decoder(h)

        # Ensure correct length
        if x_rec.size(2) != self.sequence_length:
            x_rec = F.interpolate(x_rec, size=self.sequence_length, mode='linear', align_corners=False)

        return x_rec


class PatternVAE(nn.Module):
    """
    Variational Autoencoder for pattern detection.

    Learns unsupervised representations of price patterns and
    detects anomalies via reconstruction error.
    """

    def __init__(
        self,
        input_channels: int = 7,
        sequence_length: int = 64,
        latent_dim: int = 32,
        hidden_dims: List[int] = None
    ):
        super().__init__()

        self.encoder = PatternEncoder(input_channels, sequence_length, latent_dim, hidden_dims)
        self.decoder = PatternDecoder(latent_dim, input_channels, sequence_length, hidden_dims)

        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input sequence (B, C, L)

        Returns:
            (reconstruction, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)

        return x_rec, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent representation (deterministic)."""
        mu, _ = self.encoder(x)
        return mu

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input."""
        z = self.encode(x)
        return self.decoder(z)

    def sample(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        """Sample from latent space."""
        z = torch.randn(n_samples, self.latent_dim).to(device)
        return self.decoder(z)


@dataclass
class PatternDetectionResult:
    """Results from pattern detection."""
    anomaly_scores: np.ndarray  # Reconstruction error for each sample
    is_anomaly: np.ndarray  # Binary anomaly flags
    latent_embeddings: np.ndarray  # Latent representations
    reconstructions: np.ndarray  # Reconstructed sequences
    threshold: float  # Anomaly threshold
    n_anomalies: int  # Number of detected anomalies


class PatternDetector:
    """
    Pattern detector using trained VAE.

    Detects:
    - Anomalous price patterns (reconstruction error)
    - Recurring patterns (clustering in latent space)
    - Pattern transitions (latent space dynamics)
    """

    def __init__(
        self,
        model: PatternVAE,
        anomaly_threshold: float = 2.0,
        device: str = "cpu"
    ):
        """
        Initialize pattern detector.

        Args:
            model: Trained PatternVAE model
            anomaly_threshold: Threshold in standard deviations for anomaly detection
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.anomaly_threshold = anomaly_threshold
        self.device = device

        self.baseline_error: Optional[float] = None
        self.baseline_std: Optional[float] = None

    def calibrate(self, normal_data: torch.Tensor):
        """
        Calibrate anomaly detector on normal data.

        Args:
            normal_data: Normal price sequences (N, C, L)
        """
        self.model.eval()

        with torch.no_grad():
            normal_data = normal_data.to(self.device)
            reconstructions, _, _ = self.model(normal_data)

            # Calculate reconstruction errors
            errors = F.mse_loss(reconstructions, normal_data, reduction='none')
            errors = errors.mean(dim=(1, 2))  # Per-sample error

            # Baseline statistics
            self.baseline_error = errors.mean().item()
            self.baseline_std = errors.std().item()

        logger.info(
            f"Calibrated anomaly detector: "
            f"baseline_error={self.baseline_error:.6f}, "
            f"std={self.baseline_std:.6f}"
        )

    def detect_patterns(self, data: torch.Tensor) -> PatternDetectionResult:
        """
        Detect patterns and anomalies in data.

        Args:
            data: Price sequences (N, C, L)

        Returns:
            PatternDetectionResult with anomaly scores and embeddings
        """
        if self.baseline_error is None:
            raise RuntimeError("Must calibrate detector before detection")

        self.model.eval()

        with torch.no_grad():
            data = data.to(self.device)
            reconstructions, mu, _ = self.model(data)

            # Calculate reconstruction errors
            errors = F.mse_loss(reconstructions, data, reduction='none')
            errors = errors.mean(dim=(1, 2))  # Per-sample error

            # Normalize by baseline
            anomaly_scores = (errors - self.baseline_error) / (self.baseline_std + 1e-8)

            # Detect anomalies
            threshold = self.anomaly_threshold
            is_anomaly = anomaly_scores > threshold

            # Convert to numpy
            anomaly_scores_np = anomaly_scores.cpu().numpy()
            is_anomaly_np = is_anomaly.cpu().numpy()
            latent_np = mu.cpu().numpy()
            reconstructions_np = reconstructions.cpu().numpy()

        result = PatternDetectionResult(
            anomaly_scores=anomaly_scores_np,
            is_anomaly=is_anomaly_np,
            latent_embeddings=latent_np,
            reconstructions=reconstructions_np,
            threshold=threshold,
            n_anomalies=int(is_anomaly_np.sum())
        )

        logger.info(
            f"Pattern detection: {result.n_anomalies}/{len(data)} anomalies "
            f"({result.n_anomalies/len(data)*100:.1f}%)"
        )

        return result

    def find_similar_patterns(
        self,
        query_pattern: torch.Tensor,
        pattern_database: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar patterns in database.

        Args:
            query_pattern: Query pattern (1, C, L)
            pattern_database: Database of patterns (N, C, L)
            top_k: Number of similar patterns to return

        Returns:
            (indices, distances) of top-k most similar patterns
        """
        self.model.eval()

        with torch.no_grad():
            # Encode query
            query_z = self.model.encode(query_pattern.to(self.device))

            # Encode database
            db_z = self.model.encode(pattern_database.to(self.device))

            # Calculate distances
            distances = torch.cdist(query_z, db_z, p=2).squeeze(0)

            # Get top-k
            top_distances, top_indices = torch.topk(distances, k=top_k, largest=False)

        return top_indices.cpu().numpy(), top_distances.cpu().numpy()

    def cluster_patterns(
        self,
        data: torch.Tensor,
        n_clusters: int = 10
    ) -> np.ndarray:
        """
        Cluster patterns in latent space.

        Args:
            data: Price sequences (N, C, L)
            n_clusters: Number of clusters

        Returns:
            Cluster labels (N,)
        """
        from sklearn.cluster import KMeans

        # Encode to latent space
        with torch.no_grad():
            latent = self.model.encode(data.to(self.device))
            latent_np = latent.cpu().numpy()

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(latent_np)

        logger.info(f"Clustered {len(data)} patterns into {n_clusters} groups")

        return labels


def train_pattern_vae(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    input_channels: int = 7,
    sequence_length: int = 64,
    latent_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu"
) -> PatternVAE:
    """
    Train pattern VAE on price data.

    Args:
        train_data: Training sequences (N, C, L)
        val_data: Validation sequences (M, C, L)
        input_channels: Number of input channels
        sequence_length: Sequence length
        latent_dim: Latent dimension
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device for training

    Returns:
        Trained PatternVAE model
    """
    from torch.utils.data import TensorDataset, DataLoader

    # Create model
    model = PatternVAE(input_channels, sequence_length, latent_dim)
    model = model.to(device)

    # Create dataloaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    logger.info(f"Training pattern VAE: {epochs} epochs, batch_size={batch_size}")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            # Forward
            x_rec, mu, logvar = model(x)

            # Loss: reconstruction + KL divergence
            recon_loss = F.mse_loss(x_rec, x, reduction='mean')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

            loss = recon_loss + 0.01 * kl_loss  # KL weight

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)

                x_rec, mu, logvar = model(x)

                recon_loss = F.mse_loss(x_rec, x, reduction='mean')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

                loss = recon_loss + 0.01 * kl_loss

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Log
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    logger.info(f"Training complete: best_val_loss={best_val_loss:.6f}")

    return model
