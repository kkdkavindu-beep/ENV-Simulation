"""
Fatigue System — Normalized to Endurance
"""
import numpy as np
from config import *
from animal_model import (fatigue, traits_shared, M_F, alive)

def recover_fatigue(alive_idx: np.ndarray):
    """Fatigue recovery: -0.2 per tick (2.0/sec)"""
    fatigue[alive_idx] = np.maximum(0.0, fatigue[alive_idx] - FATIGUE_RECOVERY)

def compute_M_F(alive_idx: np.ndarray):
    """Compute fatigue multiplier M(F) = clamp((F / (0.3*endurance))^1.5, 0.01, 1.0)"""
    endurance = traits_shared[alive_idx, T_ENDURANCE]
    threshold = FATIGUE_THRESHOLD_FRAC * endurance
    M_F[alive_idx] = np.clip(
        (fatigue[alive_idx] / threshold) ** FATIGUE_EXPONENT,
        FATIGUE_MIN_MULT, 1.0
    )

def accumulate_fatigue(fatigue_delta: np.ndarray, alive_idx: np.ndarray):
    """Add accumulated fatigue and clamp to endurance."""
    fatigue[alive_idx] += fatigue_delta[alive_idx]
    np.clip(fatigue[alive_idx], 0.0, traits_shared[alive_idx, T_ENDURANCE], out=fatigue[alive_idx])

# Pre-computed fatigue costs for actions
def get_move_fatigue(dx: float, dy: float) -> float:
    return MOVE_FATIGUE_FACTOR * (abs(dx) + abs(dy))

def get_attack_fatigue() -> float:
    return ATTACK_FATIGUE

def get_detection_fatigue(R: float) -> float:
    return DETECTION_FATIGUE_LARGE if R > 50 else DETECTION_FATIGUE_SMALL