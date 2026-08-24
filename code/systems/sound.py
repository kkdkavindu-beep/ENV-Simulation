"""
Sound System — Fixed-Array Inbox (No Heap)
"""
import numpy as np
import math
from config import *
from animal_model import (x_pos, y_pos, alive, species, animal_ids, id_to_slot,
                          animal_grid, traits_shared, M_F, infant_factor,
                          sound_inbox)

# ── Sound Emission ───────────────────────────────────────────────────────
def compute_S_emit(slot: int, str_n: int) -> float:
    """Effective emitted strength with fatigue and infant scaling."""
    snd_pwr = float(traits_shared[slot, T_SND_PWR])
    return (str_n / 255.0) * snd_pwr * float(M_F[slot]) * float(infant_factor[slot])

# ── Attenuation ──────────────────────────────────────────────────────────
def attenuate(S_emit: float, distance: float) -> float:
    return S_emit * math.exp(-ATTENUATION_K * distance)

# ── Fixed-Array Inbox Insertion ──────────────────────────────────────────
def insert_sound_fixed(receiver_slot: int, strength: float,
                       sig_type: int, sender_slot: int) -> None:
    """
    Insert into fixed-size sorted array (top 4).
    Insertion sort: O(4) = constant time, no allocation.
    """
    arr = sound_inbox[receiver_slot]  # (4, 3)
    
    # Find insertion point (strongest first)
    insert_pos = 4
    for i in range(4):
        if arr[i, 0] == 0 or strength > arr[i, 0]:
            insert_pos = i
            break
    
    if insert_pos == 4:
        return  # weaker than all 4 stored
    
    # Shift weaker entries down
    for i in range(3, insert_pos, -1):
        arr[i, 0] = arr[i-1, 0]
        arr[i, 1] = arr[i-1, 1]
        arr[i, 2] = arr[i-1, 2]
    
    # Insert new
    arr[insert_pos, 0] = strength
    arr[insert_pos, 1] = float(sig_type)
    arr[insert_pos, 2] = float(sender_slot)

# ── Broadcast ────────────────────────────────────────────────────────────
def broadcast_sounds_vectorized(
    emit_slots: np.ndarray,
    signal_type_arr: np.ndarray,
    signal_str_arr: np.ndarray,
    x_pos_arr, y_pos_arr,
    animal_grid_arr,
    traits_shared_arr,
    M_F_arr, infant_factor_arr,
    species_arr, alive_arr
) -> None:
    """
    One-pass broadcast: all emitters → all receivers.
    Uses pre-allocated sound_inbox.
    """
    # Reset inbox
    sound_inbox.fill(0.0)
    
    # Global minimum sensitivity (conservative radius cap)
    alive_mask = alive_arr
    if not np.any(alive_mask):
        return
    s_min = float(traits_shared_arr[alive_mask, T_SND_SENS].min())
    
    for slot in emit_slots:
        if not alive_arr[slot]:
            continue
        
        str_n = int(signal_str_arr[slot])
        if str_n == 0:
            continue
        
        S_emit = compute_S_emit(int(slot), str_n)
        if S_emit <= s_min:
            continue
        
        sig_type = int(signal_type_arr[slot])
        
        # Maximum range
        d_max = (math.log(S_emit) - math.log(s_min)) / ATTENUATION_K
        
        # Candidate receivers from grid
        cx, cy = float(x_pos_arr[slot]), float(y_pos_arr[slot])
        candidates = get_ids_in_radius(animal_grid_arr, cx, cy, d_max)
        
        for rid in candidates:
            rslot = id_to_slot.get(rid)
            if rslot is None or rslot == slot or not alive_arr[rslot]:
                continue
            
            dx = float(x_pos_arr[rslot]) - cx
            dy = float(y_pos_arr[rslot]) - cy
            d = math.sqrt(dx*dx + dy*dy)
            if d == 0:
                d = 0.001
            
            S_recv = attenuate(S_emit, d)
            threshold = float(traits_shared_arr[rslot, T_SND_SENS])
            
            if S_recv >= threshold:
                insert_sound_fixed(rslot, S_recv, sig_type, slot)

# ── Grid Radius Query ────────────────────────────────────────────────────
def get_ids_in_radius(grid, cx: float, cy: float, radius: float) -> list:
    """Get all entity IDs within radius of (cx, cy)."""
    candidates = []
    r2 = radius * radius
    cell_size = CELL_SIZE
    row_min = max(0, int((cy - radius) // cell_size))
    row_max = min(GRID_DIM - 1, int((cy + radius) // cell_size))
    
    for row in range(row_min, row_max + 1):
        row_y_min = row * cell_size
        row_y_max = (row + 1) * cell_size
        nearest_y = max(row_y_min, min(row_y_max, cy))
        dy = abs(nearest_y - cy)
        if dy > radius:
            continue
        dx_span = math.sqrt(r2 - dy * dy)
        col_min = max(0, int((cx - dx_span) // cell_size))
        col_max = min(GRID_DIM - 1, int((cx + dx_span) // cell_size))
        
        for col in range(col_min, col_max + 1):
            candidates.extend(grid[row][col])
    
    return candidates

# ── Carcass Scent ────────────────────────────────────────────────────────
def add_carcass_scent_to_inbox(carcass_grid_arr, x_pos_arr, y_pos_arr,
                               sound_inbox_arr, alive_arr, species_arr,
                               traits_shared_arr):
    """Add carcass scent (type 15) to nearby carnivores' inboxes."""
    from animal_model import carc_x, carc_y, carc_ep, carc_alive
    
    carc_indices = np.where(carc_alive)[0]
    for idx in carc_indices:
        if carc_ep[idx] <= 0:
            continue
        
        cx, cy = float(carc_x[idx]), float(carc_y[idx])
        S_emit = carc_ep[idx] * 0.1  # scent strength proportional to EP
        
        # Only carnivores care about carcass scent
        candidates = get_ids_in_radius(carcass_grid_arr, cx, cy, 300.0)
        for rid in candidates:
            rslot = id_to_slot.get(rid)
            if rslot is None or not alive_arr[rslot] or species_arr[rslot]:
                continue  # only carnivores (species=False)
            
            dx = float(x_pos_arr[rslot]) - cx
            dy = float(y_pos_arr[rslot]) - cy
            d = math.sqrt(dx*dx + dy*dy)
            if d == 0: d = 0.001
            
            S_recv = attenuate(S_emit, d)
            threshold = float(traits_shared_arr[rslot, T_SND_SENS])
            
            if S_recv >= threshold:
                insert_sound_fixed(rslot, S_recv, CARCASS_SCENT_TYPE, -1)

# ── Build Sound Inputs ───────────────────────────────────────────────────
def build_sound_inputs_batch(slots: np.ndarray) -> np.ndarray:
    """
    Extract top-4 sounds from fixed inbox for batch of slots.
    Returns (N, 8) float32: [type1, type2, type3, type4, str1, str2, str3, str4]
    """
    if len(slots) == 0:
        return np.zeros((0, 8), dtype=np.float32)
    
    # sound_inbox[slots] is (N, 4, 3) — already sorted strongest-first
    inbox_batch = sound_inbox[slots]  # (N, 4, 3)
    
    out = np.zeros((len(slots), 8), dtype=np.float32)
    out[:, 0:4] = inbox_batch[:, :, 1]  # signal_type
    out[:, 4:8] = inbox_batch[:, :, 0]  # strength
    
    return out