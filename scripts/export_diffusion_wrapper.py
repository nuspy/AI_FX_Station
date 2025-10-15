import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, 'D:/Projects/ForexGPT/src')

from forex_diffusion.train.loop import ForexDiffusionLit

# Create an improved wrapper class with candles support
class DiffusionModelWrapper:
    """Wrapper to make ForexDiffusionLit compatible with inference pipeline."""

    def __init__(self, lightning_model, metadata):
        self.model = lightning_model
        self.model.eval()
        self.metadata = metadata
        self.n_samples = 100  # Number of samples for prediction

        # Extract model configuration
        self.patch_len = metadata.get('patch_len', 16)
        self.channel_order = metadata.get('channel_order', ['open', 'high', 'low', 'close', 'volume', 'hour_sin', 'hour_cos'])
        self.mu = np.array(metadata.get('mu', []))
        self.sigma = np.array(metadata.get('sigma', []))

    def _build_patch_from_candles(self, candles_df):
        """
        Build patch tensor from raw candles dataframe.

        Args:
            candles_df: DataFrame with OHLCV + time features

        Returns:
            patch: Tensor (1, C, L) ready for model
        """
        # Extract last patch_len candles
        if len(candles_df) < self.patch_len:
            raise ValueError(f"Need at least {self.patch_len} candles, got {len(candles_df)}")

        patch_df = candles_df.tail(self.patch_len).copy()

        # Add time features if not present
        if 'hour_sin' not in patch_df.columns or 'hour_cos' not in patch_df.columns:
            ts = pd.to_datetime(patch_df['ts_utc'], unit='ms', utc=True)
            hours = ts.dt.hour
            patch_df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
            patch_df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

        # Ensure volume column
        if 'volume' not in patch_df.columns:
            patch_df['volume'] = 0.0

        # Extract values in channel order
        values = []
        for channel in self.channel_order:
            if channel in patch_df.columns:
                values.append(patch_df[channel].values.astype(np.float32))
            else:
                values.append(np.zeros(len(patch_df), dtype=np.float32))

        # Stack to (C, L) and normalize
        patch_array = np.stack(values, axis=0)  # (C, L)

        # Apply standardization (same as training)
        if len(self.mu) > 0 and len(self.sigma) > 0:
            for i in range(patch_array.shape[0]):
                if i < len(self.mu) and i < len(self.sigma):
                    if self.sigma[i] > 0:
                        patch_array[i] = (patch_array[i] - self.mu[i]) / self.sigma[i]

        # Convert to tensor and add batch dimension
        patch_tensor = torch.from_numpy(patch_array).unsqueeze(0)  # (1, C, L)

        return patch_tensor

    def predict(self, X, candles_df=None):
        """
        Predict method compatible with sklearn interface.

        Args:
            X: Input features (B, num_features) - ignored for diffusion models
            candles_df: Raw OHLCV candles dataframe (required)

        Returns:
            predictions: Array of predicted close prices (B,)
        """
        if candles_df is None:
            raise ValueError("Diffusion model requires candles_df for prediction")

        with torch.no_grad():
            # Build patch from candles
            x_patch = self._build_patch_from_candles(candles_df)

            # Encode to latent space
            mu, logvar = self.model.vae.encode(x_patch)
            std = torch.exp(0.5 * logvar)

            # Generate multiple samples and take median
            samples = []
            for _ in range(self.n_samples):
                eps = torch.randn_like(std)
                z = mu + eps * std
                x_rec = self.model.vae.decode(z)

                # Extract close price (denormalize)
                close_idx = self.channel_order.index('close') if 'close' in self.channel_order else 3
                close_normalized = x_rec[0, close_idx, -1].item()

                # Denormalize using mu/sigma
                if close_idx < len(self.mu) and close_idx < len(self.sigma):
                    close_pred = close_normalized * self.sigma[close_idx] + self.mu[close_idx]
                else:
                    close_pred = close_normalized

                samples.append(close_pred)

            # Compute median prediction
            prediction = np.median(samples)

            return np.array([prediction])

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model = self.model.to(device)
        return self

if __name__ == "__main__":
    # Load checkpoint and metadata
    ckpt_path = Path('D:/Projects/ForexGPT/artifacts/lightning/EURUSD-1m-epoch=01-val/loss=1.8001.ckpt')
    meta_path = ckpt_path.with_suffix('.ckpt.meta.json')
    output_pt_path = ckpt_path.parent / ckpt_path.stem.replace('loss=', 'EURUSD-1m-best-')
    output_pt_path = output_pt_path.with_suffix('.pt')
    output_meta_pt = output_pt_path.with_suffix('.pt.meta.json')

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)

    print(f"Loading metadata: {meta_path}")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # Recreate model and wrap it
    print("Creating improved wrapped model...")
    base_model = ForexDiffusionLit()
    base_model.load_state_dict(ckpt['state_dict'])
    base_model.eval()

    # Create wrapper
    wrapped_model = DiffusionModelWrapper(base_model, metadata)

    # Build export
    export_data = {
        'model': wrapped_model,
        'state_dict': ckpt['state_dict'],

        'metadata': {
            'symbol': metadata['symbol'],
            'timeframe': metadata['timeframe'],
            'horizon_bars': metadata['horizon_bars'],
            'patch_len': metadata['patch_len'],
            'channel_order': metadata['channel_order'],
            'mu': metadata['mu'],
            'sigma': metadata['sigma'],
            'indicator_tfs': metadata['indicator_tfs'],
            'warmup_bars': metadata['warmup_bars'],
            'rv_window': metadata['rv_window'],
            'lightning': {
                'epoch': ckpt['epoch'],
                'global_step': ckpt['global_step'],
                'pytorch_lightning_version': ckpt.get('pytorch-lightning_version', 'unknown'),
            }
        },

        'standardizer': {
            'mu': metadata['mu'],
            'sigma': metadata['sigma'],
            'channel_order': metadata['channel_order']
        },

        'features': [],
        'model_type': 'vae_diffusion',
        'hyper_parameters': ckpt.get('hyper_parameters', {}),
        'channel_order': metadata['channel_order'],
        'mu': metadata['mu'],
        'sigma': metadata['sigma'],
    }

    # Extract features
    features = []
    for indicator, tfs in metadata['indicator_tfs'].items():
        for tf in tfs:
            features.append(f"{indicator}_{tf}")
    export_data['features'] = features

    print(f"\nExporting improved wrapper to: {output_pt_path}")
    print(f"  - Patch length: {wrapped_model.patch_len}")
    print(f"  - Channels: {wrapped_model.channel_order}")

    # Save
    torch.save(export_data, str(output_pt_path))
    print(f"[OK] Saved wrapped .pt file")

    with open(output_meta_pt, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved metadata")

    print(f"\n[SUCCESS] Improved diffusion wrapper exported!")
    print(f"  - Now requires candles_df parameter")
    print(f"  - Builds patches from OHLCV data")
    print(f"  - Applies proper normalization/denormalization")
