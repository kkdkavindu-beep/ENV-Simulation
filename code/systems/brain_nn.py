"""
Neural Network — Batched Einsum Forward Pass
"""
import numpy as np
from .config import *
from .rng import get_rng

# Weight tensors (imported from animal_model to avoid circular import)
W1_h = W2_h = W3_h = b1_h = b2_h = b3_h = None
W1_c = W2_c = W3_c = b1_c = b2_c = b3_c = None

def set_weight_tensors(w1_h, w2_h, w3_h, b1_h_, b2_h_, b3_h_,
                       w1_c, w2_c, w3_c, b1_c_, b2_c_, b3_c_):
    """Set global references to weight tensors from animal_model."""
    global W1_h, W2_h, W3_h, b1_h, b2_h, b3_h
    global W1_c, W2_c, W3_c, b1_c, b2_c, b3_c
    W1_h, W2_h, W3_h, b1_h, b2_h, b3_h = w1_h, w2_h, w3_h, b1_h_, b2_h_, b3_h_
    W1_c, W2_c, W3_c, b1_c, b2_c, b3_c = w1_c, w2_c, w3_c, b1_c_, b2_c_, b3_c_

# ── Activation Functions ─────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def int_map(z: np.ndarray, hi: int) -> np.ndarray:
    """Map tanh output → integer [0, hi]."""
    return np.round((np.tanh(z) + 1.0) / 2.0 * hi).astype(np.int32)

# ── Forward Pass ─────────────────────────────────────────────────────────
def forward_batch(idx: np.ndarray, inputs: np.ndarray,
                  W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray,
                  W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Batched forward pass for N animals.
    Returns (N, OUTPUT_SIZE) raw logits.
    """
    W1_n = W1[idx]
    b1_n = b1[idx]
    W2_n = W2[idx]
    b2_n = b2[idx]
    W3_n = W3[idx]
    b3_n = b3[idx]
    
    H1 = np.tanh(np.einsum('nij,nj->ni', W1_n, inputs) + b1_n)
    H2 = np.tanh(np.einsum('nij,nj->ni', W2_n, H1)      + b2_n)
    Z3 =          np.einsum('nij,nj->ni', W3_n, H2)     + b3_n
    
    return Z3

# ── Input Building ───────────────────────────────────────────────────────
def build_inputs(idx: np.ndarray,
                 scan_results: dict,
                 sound_inputs: np.ndarray,
                 mem_read_idx: np.ndarray) -> np.ndarray:
    """
    Build (N, INPUT_SIZE) input matrix for batch of animals.
    Fully vectorized.
    """
    N = len(idx)
    inputs = np.zeros((N, INPUT_SIZE), dtype=np.float32)
    
    # Detection inputs (0-15): 8 combined + 8 obstacle
    for i, slot in enumerate(idx):
        if slot in scan_results:
            inputs[i, 0:16] = scan_results[slot]
    
    # Sound inputs (16-23): 4 types + 4 strengths
    if sound_inputs is not None and len(sound_inputs) == N:
        inputs[:, 16:24] = sound_inputs
    
    # Internal state (24-27): energy, health, age_norm, fatigue_norm
    from .animal_model import energy, health, age, fatigue, traits_shared
    inputs[:, 24] = energy[idx]
    inputs[:, 25] = health[idx]
    inputs[:, 26] = age[idx].astype(np.float32) / traits_shared[idx, T_LIFESPAN]
    inputs[:, 27] = fatigue[idx] / traits_shared[idx, T_ENDURANCE]
    
    # Memory inputs (28-31): 4 selected cells
    from .animal_model import memory
    mem_vals = memory[idx]  # (N, 16) uint8
    row_idx = np.arange(N)[:, None]
    inputs[:, 28:32] = mem_vals[row_idx, mem_read_idx].astype(np.float32) / 255.0
    
    return inputs

# ── Output Decoding ──────────────────────────────────────────────────────
def decode_outputs(Z3: np.ndarray, is_herb: bool) -> dict:
    """
    Decode (N, OUTPUT_SIZE) raw logits into dict of output arrays.
    """
    return {
        "move_x":         np.tanh(Z3[:, 0]),
        "move_y":         np.tanh(Z3[:, 1]),
        "detect_active":  np.round(sigmoid(Z3[:, 2])).astype(np.bool_),
        "detect_dir":     int_map(Z3[:, 3], NUM_DIR_BINS - 1),
        "detect_range":   int_map(Z3[:, 4], 255),
        "detect_angle":   int_map(Z3[:, 5], ANGLE_LEVELS - 1),
        "target_lock":    int_map(Z3[:, 6], 255),
        "action7":        int_map(Z3[:, 7], 255),
        "feed":           np.round(sigmoid(Z3[:, 8])).astype(np.bool_),
        "mate_seek":      np.round(sigmoid(Z3[:, 9])).astype(np.bool_),
        "energy_transfer":np.round(sigmoid(Z3[:, 10])).astype(np.bool_),
        "signal_active":  np.round(sigmoid(Z3[:, 11])).astype(np.bool_),
        "signal_type":    int_map(Z3[:, 12], 15),
        "signal_str":     int_map(Z3[:, 13], 255),
        "mem_write_idx":  int_map(Z3[:, 14], 15),
        "mem_write_val":  int_map(Z3[:, 15], 255),
    }

# ── Detection Parameter Decoding ─────────────────────────────────────────
def decode_detection_params(dir_n: np.ndarray, range_n: np.ndarray, angle_n: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized decode for batch."""
    theta = (2 * np.pi * dir_n.astype(np.float32) / NUM_DIR_BINS) - np.pi
    R = (2.0 * R_MAX * range_n.astype(np.float32)) / 255.0
    alpha = ANGLE_MAP[angle_n]
    return theta, R, alpha

# ── Memory Read Index ────────────────────────────────────────────────────
# Track last 4 written indices per animal for reading
mem_read_history = np.zeros((MAX_ANIMALS, MEMORY_READ_CELLS), dtype=np.int8)
mem_read_ptr = np.zeros(MAX_ANIMALS, dtype=np.int8)

def update_memory_read_idx(slot: int, write_idx: int):
    """Call after memory write to update read history."""
    ptr = mem_read_ptr[slot]
    mem_read_history[slot, ptr] = write_idx
    mem_read_ptr[slot] = (ptr + 1) % MEMORY_READ_CELLS

def get_memory_read_idx(slots: np.ndarray) -> np.ndarray:
    """Return (N, 4) read indices for given slots."""
    return mem_read_history[slots].copy()

# ── Main NN Tick ─────────────────────────────────────────────────────────
def run_nn_tick(herb_idx: np.ndarray, carn_idx: np.ndarray,
                scan_results: dict,
                sound_inputs_herb: np.ndarray,
                sound_inputs_carn: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Run both species batches. Returns (herb_Z3, carn_Z3)."""
    
    mem_read_herb = get_memory_read_idx(herb_idx) if len(herb_idx) > 0 else np.zeros((0, 4), dtype=np.int8)
    mem_read_carn = get_memory_read_idx(carn_idx) if len(carn_idx) > 0 else np.zeros((0, 4), dtype=np.int8)
    
    inputs_h = build_inputs(herb_idx, scan_results, sound_inputs_herb, mem_read_herb) if len(herb_idx) > 0 else np.zeros((0, INPUT_SIZE), dtype=np.float32)
    inputs_c = build_inputs(carn_idx, scan_results, sound_inputs_carn, mem_read_carn) if len(carn_idx) > 0 else np.zeros((0, INPUT_SIZE), dtype=np.float32)
    
    Z3_h = forward_batch(herb_idx, inputs_h, W1_h, b1_h, W2_h, b2_h, W3_h, b3_h) if len(herb_idx) > 0 else np.zeros((0, OUTPUT_SIZE), dtype=np.float32)
    Z3_c = forward_batch(carn_idx, inputs_c, W1_c, b1_c, W2_c, b2_c, W3_c, b3_c) if len(carn_idx) > 0 else np.zeros((0, OUTPUT_SIZE), dtype=np.float32)
    
    return Z3_h, Z3_c

# ── GPU Acceleration (Optional) ──────────────────────────────────────────
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def forward_batch_gpu(idx, inputs, W1, b1, W2, b2, W3, b3):
    if not HAS_CUPY:
        return forward_batch(idx, inputs, W1, b1, W2, b2, W3, b3)
    
    inputs_g = cp.asarray(inputs)
    W1_g = cp.asarray(W1[idx])
    b1_g = cp.asarray(b1[idx])
    W2_g = cp.asarray(W2[idx])
    b2_g = cp.asarray(b2[idx])
    W3_g = cp.asarray(W3[idx])
    b3_g = cp.asarray(b3[idx])
    
    H1 = cp.tanh(cp.einsum('nij,nj->ni', W1_g, inputs_g) + b1_g)
    H2 = cp.tanh(cp.einsum('nij,nj->ni', W2_g, H1)       + b2_g)
    Z3 =          cp.einsum('nij,nj->ni', W3_g, H2)      + b3_g
    
    return cp.asnumpy(Z3)