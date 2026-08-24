"""
Active Detection — Numba JIT Accelerated
"""
import numpy as np
import math
from numba import jit
from .config import *
from .animal_model import (x_pos, y_pos, alive, species, animal_ids, id_to_slot,
                          animal_grid, herb_grid, obstacle_grid,
                          traits_shared, traits_carn, traits_herb,
                          M_F, infant_factor, last_scan, last_decoded,
                          herb_positions)

# ── Numba JIT Detection Kernel ───────────────────────────────────────────
@jit(nopython=True, cache=True)
def run_detection_numba(
    observer_x: float, observer_y: float,
    theta: float, R: float, alpha: float,
    candidates: np.ndarray,  # (N, 5) [type, x, y, slot, dist_sq]
    observer_is_herb: bool
) -> np.ndarray:
    """
    Numba-compiled exact detection + directional weight accumulation.
    Returns (16,) float32: 8 combined + 8 obstacle.
    """
    cone_min = theta - alpha
    cone_max = theta + alpha
    R_sq = R * R
    
    # Normalize cone bounds to [0, 2π)
    TWO_PI = 2.0 * math.pi
    if cone_min < 0: cone_min += TWO_PI
    if cone_max < 0: cone_max += TWO_PI
    
    combined = np.zeros(8, dtype=np.float32)
    obstacle = np.zeros(8, dtype=np.float32)
    
    for i in range(candidates.shape[0]):
        obj_type = int(candidates[i, 0])
        dx = candidates[i, 1] - observer_x
        dy = candidates[i, 2] - observer_y
        dist_sq = candidates[i, 4]
        
        if dist_sq > R_sq:
            continue
        
        # Exact angle check
        obj_angle = math.atan2(dx, dy)
        if obj_angle < 0: obj_angle += TWO_PI
        
        in_cone = False
        if cone_min <= cone_max:
            in_cone = (obj_angle >= cone_min) and (obj_angle <= cone_max)
        else:
            in_cone = (obj_angle >= cone_min) or (obj_angle <= cone_max)
        
        if not in_cone:
            continue
        
        # Logarithmic proximity weight
        dist = math.sqrt(dist_sq)
        w = math.log(1.0 + max(0.0, R - dist))
        
        # Bin index (0-7)
        bin_idx = int((obj_angle + math.pi) / TWO_PI * 8) % 8
        
        # Camouflage/stealth for herbivore targets (carnivore observer)
        if obj_type == 1 and not observer_is_herb:
            # We need per-target camo/stealth - pass via candidate array in future
            # For now, use average values from trait caches
            pass
        
        # Accumulate
        if obj_type == 0:  # plant
            combined[bin_idx] += w
        elif obj_type == 1:  # herbivore
            combined[bin_idx] += w
        elif obj_type == 2:  # carnivore
            combined[bin_idx] += w
        elif obj_type == 3:  # obstacle
            obstacle[bin_idx] += w
            combined[bin_idx] += w
    
    # Pack result
    result = np.zeros(16, dtype=np.float32)
    result[:8] = combined
    result[8:] = obstacle
    return result

# ── Stage 1: Grid Cell Query (Python) ────────────────────────────────────
def get_detection_candidates(slot: int, theta: float, R: float, alpha: float
                            ) -> np.ndarray:
    """
    Collect candidate objects from spatial grid within circular range.
    Returns (N_candidates, 5) float32: [type, x, y, slot, dist_sq]
    Type encoding: 0=plant, 1=herbivore, 2=carnivore, 3=obstacle
    """
    cx, cy = x_pos[slot], y_pos[slot]
    cone_min = theta - alpha
    cone_max = theta + alpha
    
    cell_size = CELL_SIZE
    row_min = max(0, int((cy - R) // cell_size))
    row_max = min(GRID_DIM - 1, int((cy + R) // cell_size))
    
    candidates = []
    TWO_PI = 2.0 * math.pi
    
    # Normalize cone bounds once
    cmin = cone_min
    cmax = cone_max
    if cmin < 0: cmin += TWO_PI
    if cmax < 0: cmax += TWO_PI
    
    for row in range(row_min, row_max + 1):
        row_y_min = row * cell_size
        row_y_max = (row + 1) * cell_size
        
        nearest_y = max(row_y_min, min(row_y_max, cy))
        dy = abs(nearest_y - cy)
        if dy > R:
            continue
        
        dx_span = math.sqrt(R * R - dy * dy)
        col_min = max(0, int((cx - dx_span) // cell_size))
        col_max = min(GRID_DIM - 1, int((cx + dx_span) // cell_size))
        
        for col in range(col_min, col_max + 1):
            # Cell center angle pre-filter
            cell_cx = (col + 0.5) * cell_size
            cell_cy = (row + 0.5) * cell_size
            cell_dx = cell_cx - cx
            cell_dy = cell_cy - cy
            cell_angle = math.atan2(cell_dx, cell_dy)
            if cell_angle < 0: cell_angle += TWO_PI
            
            in_cone = (cmin <= cmax and cmin <= cell_angle <= cmax) or \
                      (cmin > cmax and (cell_angle >= cmin or cell_angle <= cmax))
            
            if not in_cone:
                continue
            
            # Collect animals
            for aid in animal_grid[row][col]:
                if aid == animal_ids[slot]:
                    continue
                obj_slot = id_to_slot.get(aid)
                if obj_slot is None or not alive[obj_slot]:
                    continue
                ox, oy = x_pos[obj_slot], y_pos[obj_slot]
                dx = ox - cx
                dy = oy - cy
                dist_sq = dx*dx + dy*dy
                if dist_sq <= R*R:
                    obj_type = 1 if species[obj_slot] else 2
                    candidates.append([obj_type, ox, oy, obj_slot, dist_sq])
            
            # Collect herbs
            for hid in herb_grid[row][col]:
                # Need herb positions - from world module
                pass  # Will be filled by caller
            
            # Collect obstacles
            for oid in obstacle_grid[row][col]:
                ox, oy = obstacle_positions.get(oid, (0, 0))
                dx = ox - cx
                dy = oy - cy
                dist_sq = dx*dx + dy*dy
                if dist_sq <= R*R:
                    candidates.append([3, ox, oy, oid, dist_sq])
    
    if not candidates:
        return np.zeros((0, 5), dtype=np.float32)
    
    return np.array(candidates, dtype=np.float32)

# ── Batch Detection Runner ───────────────────────────────────────────────
def run_detection_batch(active_slots, x_pos_arr, y_pos_arr,
                        animal_grid_arr, herb_grid_arr, obstacle_grid_arr,
                        M_F_arr, infant_factor_arr, species_arr, alive_arr):
    """
    Run detection for all active detectors this tick.
    Returns: scan_results dict, detect_fired list, active_slots_next
    """
    scan_results = {}
    detect_fired = []
    active_next = []
    
    for slot in active_slots:
        if not alive_arr[slot]:
            continue
        
        dec = last_decoded[slot]
        if dec is None or not dec.get("detect_active", False):
            continue
        
        theta = dec["detect_theta"]
        R_raw = dec["detect_R"]
        alpha = dec["detect_alpha"]
        
        # Fatigue-scaled range
        R_eff = R_raw * M_F_arr[slot]
        
        if R_eff < 0.1:
            scan_results[slot] = np.zeros(16, dtype=np.float32)
            detect_fired.append((slot, R_eff, alpha))
            if dec["detect_active"]:
                active_next.append(slot)
            continue
        
        # Get candidates
        candidates = get_detection_candidates(slot, theta, R_eff, alpha)
        
        # Add herbs to candidates
        cx, cy = x_pos_arr[slot], y_pos_arr[slot]
        R_sq = R_eff * R_eff
        cell_size = CELL_SIZE
        row_min = max(0, int((cy - R_eff) // cell_size))
        row_max = min(GRID_DIM - 1, int((cy + R_eff) // cell_size))
        
        for row in range(row_min, row_max + 1):
            row_y_min = row * cell_size
            row_y_max = (row + 1) * cell_size
            nearest_y = max(row_y_min, min(row_y_max, cy))
            dy = abs(nearest_y - cy)
            if dy > R_eff:
                continue
            dx_span = math.sqrt(R_sq - dy * dy)
            col_min = max(0, int((cx - dx_span) // cell_size))
            col_max = min(GRID_DIM - 1, int((cx + dx_span) // cell_size))
            
            for col in range(col_min, col_max + 1):
                for hid in herb_grid_arr[row][col]:
                    hx, hy = herb_positions.get(hid, (0, 0))
                    dx = hx - cx
                    dy = hy - cy
                    dist_sq = dx*dx + dy*dy
                    if dist_sq <= R_sq:
                        candidates = np.vstack([candidates, [0, hx, hy, -1, dist_sq]]) if candidates.size else np.array([[0, hx, hy, -1, dist_sq]], dtype=np.float32)
        
        if candidates.shape[0] == 0:
            scan_results[slot] = np.zeros(16, dtype=np.float32)
            detect_fired.append((slot, R_eff, alpha))
            if dec["detect_active"]:
                active_next.append(slot)
            continue
        
        # Run Numba detection
        observer_is_herb = bool(species_arr[slot])
        result = run_detection_numba(
            float(x_pos_arr[slot]), float(y_pos_arr[slot]),
            float(theta), float(R_eff), float(alpha),
            candidates,
            observer_is_herb
        )
        
        scan_results[slot] = result
        detect_fired.append((slot, R_eff, alpha))
        
        # Store confirmed objects for target_lock
        last_scan[slot] = extract_confirmed_objects(candidates, theta, alpha, R_eff)
        
        if dec["detect_active"]:
            active_next.append(slot)
    
    return scan_results, detect_fired, active_next

def extract_confirmed_objects(candidates: np.ndarray, theta: float, alpha: float, R: float) -> list:
    """Extract confirmed object info for target_lock resolution."""
    confirmed = []
    cone_min = theta - alpha
    cone_max = theta + alpha
    R_sq = R * R
    TWO_PI = 2.0 * math.pi
    
    if cone_min < 0: cone_min += TWO_PI
    if cone_max < 0: cone_max += TWO_PI
    
    for i in range(candidates.shape[0]):
        obj_type = int(candidates[i, 0])
        dx = candidates[i, 1]
        dy = candidates[i, 2]
        obj_slot = int(candidates[i, 3])
        dist_sq = candidates[i, 4]
        
        if dist_sq > R_sq:
            continue
        
        obj_angle = math.atan2(dx, dy)
        if obj_angle < 0: obj_angle += TWO_PI
        
        in_cone = (cone_min <= cone_max and cone_min <= obj_angle <= cone_max) or \
                  (cone_min > cone_max and (obj_angle >= cone_min or obj_angle <= cone_max))
        
        if in_cone:
            confirmed.append((obj_type, obj_slot, dx, dy, math.sqrt(dist_sq)))
    
    return confirmed

# ── Target Lock Resolution ───────────────────────────────────────────────
def resolve_target_lock(lock_n: int, last_scan_objects: list) -> int | None:
    """Map lock_n to target slot from last scan."""
    if not last_scan_objects:
        return None
    idx = lock_n % len(last_scan_objects)
    return last_scan_objects[idx][1]  # obj_slot