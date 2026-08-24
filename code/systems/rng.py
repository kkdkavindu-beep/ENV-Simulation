"""
Deterministic RNG — single seeded generator for reproducibility.
All randomness in the simulation flows through this.
"""
import numpy as np
from .config import DEFAULT_SEED

# Global generator
_rng: np.random.Generator = None

def init_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Initialize the global RNG with a seed."""
    global _rng
    _rng = np.random.default_rng(seed)
    return _rng

def get_rng() -> np.random.Generator:
    """Get the global RNG instance."""
    global _rng
    if _rng is None:
        _rng = np.random.default_rng(DEFAULT_SEED)
    return _rng

def reseed(seed: int) -> None:
    """Reseed the global RNG (for testing/reproducibility)."""
    global _rng
    _rng = np.random.default_rng(seed)

# Convenience functions
def random() -> float:
    return get_rng().random()

def uniform(low: float, high: float) -> float:
    return get_rng().uniform(low, high)

def normal(loc: float = 0.0, scale: float = 1.0) -> float:
    return get_rng().normal(loc, scale)

def choice(arr, p=None):
    return get_rng().choice(arr, p=p)

def integers(low: int, high: int, size=None):
    return get_rng().integers(low, high, size=size)

def random_array(size) -> np.ndarray:
    return get_rng().random(size)

def choice_array(choices, size, p=None) -> np.ndarray:
    return get_rng().choice(choices, size=size, p=p)