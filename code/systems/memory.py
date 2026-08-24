"""
Memory System — Integer Blend, 16 Cells
"""
import numpy as np
from config import *
from animal_model import memory

# Memory read history (track last 4 written indices)
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

def blend_memory(alive_idx: np.ndarray, write_idx: np.ndarray, write_val: np.ndarray):
    """
    Integer blend: memory = (memory * 7 + new * 3 + 5) // 10
    Equivalent to 0.7 * old + 0.3 * new with rounding.
    """
    for i, slot in enumerate(alive_idx):
        idx = write_idx[i]
        val = write_val[i]
        old = memory[slot, idx]
        # Integer arithmetic: (old * 7 + val * 3 + 5) // 10
        memory[slot, idx] = np.clip((old * MEMORY_BLEND_OLD + val * MEMORY_BLEND_NEW + 5) // MEMORY_BLEND_DIV, 0, 255).astype(np.uint8)
        update_memory_read_idx(slot, idx)

def clear_memory(slot: int):
    """Clear memory on death/birth."""
    memory[slot].fill(0)