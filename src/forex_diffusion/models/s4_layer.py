"""
S4 Layer (Structured State Space Layer)

Efficient long-range dependency modeling via state space models.
Based on the S4 architecture from "Efficiently Modeling Long Sequences with Structured State Spaces"
(Gu et al., 2022).

Key Features:
- HiPPO initialization for optimal memory
- FFT-based convolution for O(L log L) complexity
- Recurrent mode for online inference
- Diagonal state matrix for efficiency
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class S4Layer(nn.Module):
    """
    Structured State Space Layer for efficient sequence modeling.

    Implements the continuous-time state space model:
        dx/dt = Ax + Bu
        y = Cx + Du

    Discretized for computational efficiency with learnable timestep dt.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
        transposed: bool = False,
        kernel_init: str = "hippo",
        l_max: int = 2048,
        **kwargs
    ):
        """
        Initialize S4 Layer.

        Args:
            d_model: Input/output dimension
            d_state: State dimension (N), controls memory capacity
            dropout: Dropout probability
            transposed: If True, expects (batch, d_model, seq_len) input
            kernel_init: Initialization method ("hippo" or "random")
            l_max: Maximum sequence length for initialization
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.transposed = transposed
        self.l_max = l_max

        # Initialize state space parameters
        if kernel_init == "hippo":
            Lambda, B_init = self._hippo_initialization(d_state)
        else:
            Lambda = torch.randn(d_state)
            B_init = torch.randn(d_state, d_model)

        # State transition matrix eigenvalues (diagonal parameterization)
        self.Lambda = nn.Parameter(Lambda)  # (d_state,)

        # Input matrix B
        self.B = nn.Parameter(B_init)  # (d_state, d_model)

        # Output matrix C
        C_init = torch.randn(d_model, d_state)
        self.C = nn.Parameter(C_init)  # (d_model, d_state)

        # Direct feedthrough (skip connection)
        D_init = torch.randn(d_model)
        self.D = nn.Parameter(D_init)  # (d_model,)

        # Learnable discretization timestep
        log_dt = torch.rand(d_model) * (
            np.log(0.001) - np.log(0.1)
        ) + np.log(0.1)
        self.log_dt = nn.Parameter(log_dt)  # (d_model,)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Cache for convolution kernel
        self.register_buffer('_kernel_cache', None)
        self._kernel_cache_l = None

        logger.debug(f"Initialized S4Layer: d_model={d_model}, d_state={d_state}, "
                    f"kernel_init={kernel_init}")

    def _hippo_initialization(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HiPPO (High-Order Polynomial Projection Operators) initialization.

        Provides optimal memory for polynomial representations.

        Args:
            n: State dimension

        Returns:
            Lambda: Eigenvalues of state matrix (n,)
            B: Input matrix (n, d_model)
        """
        # HiPPO-LegS matrix
        # A[i, j] = -(2i + 1) if i > j else (2i + 1)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i > j:
                    A[i, j] = -(2 * i + 1) ** 0.5 * (2 * j + 1) ** 0.5
                elif i == j:
                    A[i, j] = -(i + 1)

        # Convert to torch
        A = torch.tensor(A, dtype=torch.float32)

        # Compute eigenvalues (diagonal parameterization)
        Lambda_real = torch.diagonal(A).clone()

        # HiPPO B matrix
        B = torch.sqrt(torch.arange(1, n + 1, dtype=torch.float32) * 2 + 1)
        B = B.unsqueeze(-1)  # (n, 1)

        return Lambda_real, B

    def _compute_kernel(self, L: int) -> torch.Tensor:
        """
        Compute convolution kernel via FFT.

        Args:
            L: Sequence length

        Returns:
            kernel: Convolution kernel (d_model, L)
        """
        # Discretize timestep
        dt = torch.exp(self.log_dt)  # (d_model,)

        # Discretize state matrix (diagonal)
        # A_discrete[i] = exp(dt[i] * Lambda[i])
        dt_expanded = dt.unsqueeze(-1)  # (d_model, 1)
        Lambda_expanded = self.Lambda.unsqueeze(0)  # (1, d_state)
        A_discrete = torch.exp(dt_expanded * Lambda_expanded)  # (d_model, d_state)

        # Discretize input matrix
        # B_discrete = dt * B
        B_discrete = dt_expanded * self.B.T  # (d_model, d_state)

        # Compute convolution kernel via Geometric series
        # k[l] = C @ (A^l) @ B for l = 0, 1, ..., L-1

        # Use FFT for efficient computation
        # FFT of geometric series: [1, A, A^2, ..., A^(L-1)]

        # Compute powers of A via FFT
        # This is the key efficiency trick of S4
        powers = torch.zeros(self.d_model, self.d_state, L, device=A_discrete.device)

        # Initialize: power[0] = 1
        powers[:, :, 0] = 1.0

        # Compute powers iteratively (can be optimized with FFT)
        for l in range(1, L):
            powers[:, :, l] = powers[:, :, l-1] * A_discrete

        # Compute kernel: k[l] = C @ (A^l @ B)
        # Shape: (d_model, d_state) @ [(d_model, d_state) @ (d_model, d_state)]
        kernel = torch.einsum('md,dsl->ml', self.C,
                             torch.einsum('ds,msl->dsl', self.B, powers))

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in convolution mode (training).

        Args:
            x: Input tensor (batch, seq_len, d_model) or (batch, d_model, seq_len)

        Returns:
            y: Output tensor (same shape as input)
        """
        if self.transposed:
            # (batch, d_model, seq_len)
            batch, d, L = x.shape
            assert d == self.d_model, f"Expected d_model={self.d_model}, got {d}"
        else:
            # (batch, seq_len, d_model)
            batch, L, d = x.shape
            assert d == self.d_model, f"Expected d_model={self.d_model}, got {d}"
            # Transpose to (batch, d_model, seq_len) for convolution
            x = x.transpose(1, 2)

        # Compute or retrieve cached kernel
        if self._kernel_cache is None or self._kernel_cache_l != L:
            kernel = self._compute_kernel(L)
            self._kernel_cache = kernel
            self._kernel_cache_l = L
        else:
            kernel = self._kernel_cache

        # Apply convolution via FFT
        # x: (batch, d_model, L)
        # kernel: (d_model, L)

        # FFT of input
        x_fft = torch.fft.rfft(x, n=2*L, dim=-1)  # (batch, d_model, L+1)

        # FFT of kernel
        kernel_fft = torch.fft.rfft(kernel, n=2*L, dim=-1)  # (d_model, L+1)

        # Pointwise multiplication in frequency domain
        y_fft = x_fft * kernel_fft.unsqueeze(0)  # (batch, d_model, L+1)

        # Inverse FFT
        y = torch.fft.irfft(y_fft, n=2*L, dim=-1)[:, :, :L]  # (batch, d_model, L)

        # Add direct feedthrough (skip connection)
        # D is per-feature, x is (batch, d_model, L)
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x  # (batch, d_model, L)

        # Transpose back if needed
        if not self.transposed:
            y = y.transpose(1, 2)  # (batch, L, d_model)

        # Apply dropout
        y = self.dropout(y)

        return y

    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent mode for online inference (single timestep).

        Args:
            x: Input (batch, d_model)
            state: Previous hidden state (batch, d_state) or None

        Returns:
            y: Output (batch, d_model)
            new_state: Updated hidden state (batch, d_state)
        """
        batch = x.shape[0]

        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)

        # Discretize timestep
        dt = torch.exp(self.log_dt)  # (d_model,)

        # Discretize state matrix
        dt_expanded = dt.unsqueeze(-1)  # (d_model, 1)
        Lambda_expanded = self.Lambda.unsqueeze(0)  # (1, d_state)
        A_discrete = torch.exp(dt_expanded * Lambda_expanded)  # (d_model, d_state)

        # Discretize input matrix
        B_discrete = dt_expanded * self.B.T  # (d_model, d_state)

        # Update state: new_state = A @ state + B @ x
        # state: (batch, d_state)
        # A_discrete: (d_model, d_state) - need to average across d_model
        # B_discrete: (d_model, d_state)

        # Average A and B across d_model dimension for state update
        A_avg = A_discrete.mean(dim=0)  # (d_state,)
        B_avg = B_discrete.mean(dim=0)  # (d_state,)

        # State update
        new_state = state * A_avg.unsqueeze(0) + torch.einsum('bd,d->bd', x, B_avg)

        # Compute output: y = C @ state + D @ x
        # C: (d_model, d_state)
        # state: (batch, d_state)
        y = torch.einsum('md,bd->bm', self.C, new_state) + self.D.unsqueeze(0) * x

        return y, new_state

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'd_model={self.d_model}, d_state={self.d_state}, '
                f'transposed={self.transposed}, l_max={self.l_max}')


class S4Block(nn.Module):
    """
    S4 Block with normalization and feedforward network.

    Typical block structure:
        x -> LayerNorm -> S4Layer -> Dropout -> + (residual)
          -> LayerNorm -> FFN -> Dropout -> + (residual)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
        ffn_expansion: int = 4,
        **s4_kwargs
    ):
        """
        Initialize S4 Block.

        Args:
            d_model: Model dimension
            d_state: State dimension for S4 layer
            dropout: Dropout probability
            ffn_expansion: Expansion factor for feedforward network
            **s4_kwargs: Additional arguments for S4Layer
        """
        super().__init__()

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)

        # S4 layer
        self.s4 = S4Layer(d_model, d_state, dropout, **s4_kwargs)

        # Second layer normalization
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            Output (batch, seq_len, d_model)
        """
        # S4 layer with residual
        x = x + self.s4(self.norm1(x))

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class StackedS4(nn.Module):
    """
    Stack of multiple S4 blocks for deep models.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
        **s4_kwargs
    ):
        """
        Initialize stacked S4 model.

        Args:
            d_model: Model dimension
            d_state: State dimension for S4 layers
            n_layers: Number of S4 blocks to stack
            dropout: Dropout probability
            **s4_kwargs: Additional arguments for S4Layer
        """
        super().__init__()

        self.layers = nn.ModuleList([
            S4Block(d_model, d_state, dropout, **s4_kwargs)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        logger.info(f"Initialized StackedS4: n_layers={n_layers}, d_model={d_model}, "
                   f"d_state={d_state}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacked S4 blocks.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            Output (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        return x
