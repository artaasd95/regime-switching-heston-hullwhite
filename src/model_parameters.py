from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class HestonParameters:
    """Parameters for the Heston stochastic volatility model."""
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    sigma: float  # Volatility of variance
    rho: float    # Correlation between asset and variance
    v0: float     # Initial variance

@dataclass
class HullWhiteParameters:
    """Parameters for the Hull-White interest rate model."""
    alpha: float  # Mean reversion speed
    sigma_r: float  # Volatility of interest rate
    r0: float     # Initial interest rate

@dataclass
class RegimeParameters:
    """Parameters for regime switching."""
    transition_matrix: np.ndarray  # Transition probability matrix
    num_regimes: int              # Number of regimes

@dataclass
class ModelParameters:
    """Complete set of parameters for the Regime-Switching Heston-Hull-White model."""
    heston_params: List[HestonParameters]  # Heston parameters for each regime
    hull_white_params: List[HullWhiteParameters]  # Hull-White parameters for each regime
    regime_params: RegimeParameters
    spot_price: float
    time_horizon: float
    num_steps: int
    num_paths: int

    def __post_init__(self):
        """Validate parameters after initialization."""
        if len(self.heston_params) != self.regime_params.num_regimes:
            raise ValueError("Number of Heston parameter sets must match number of regimes")
        if len(self.hull_white_params) != self.regime_params.num_regimes:
            raise ValueError("Number of Hull-White parameter sets must match number of regimes")
        if self.regime_params.transition_matrix.shape != (self.regime_params.num_regimes, self.regime_params.num_regimes):
            raise ValueError("Transition matrix dimensions must match number of regimes") 