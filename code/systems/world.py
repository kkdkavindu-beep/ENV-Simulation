"""
World System — Grid, Obstacles, Herbs, Carcasses
Cell-based obstacles with local density regulation.
"""
import numpy as np
import math
from .config import *
from .rng import get_rng
from .animal_model import (obstacle_grid, herb_grid, carcass_grid, animal_grid,
                          obs_I, obs_R, obs_D, obs_x, obs_y, obs_alive,
                          herb_x, herb_y, herb_ep, herb_alive,
                          carc_x, carc_y, carc_ep, carc_age, carc_alive,
                          obstacle_positions, herb_positions,
                          grid_cell, GRID_DIM, CELL_SIZE)

# ── Obstacle R/D Computation ─────────────────────────────────────────────
def compute_R_local(row: int, col: int) -> float:
    """Compute R from local obstacle density (3x3 neighborhood)."""
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = row + dr, col + dc
            if 0 <= r < GRID_DIM and 0 <= c < GRID_DIM:
                count += len(obstacle_grid[r][c])
    
    # Density in 3x3 area (9 cells = 900 m²)
    density = count / 9.0
    
    # Piecewise linear R(density)
    if density < 0.5:
        return 0.1 + 0.3 * density  # 0.1 to 0.25
    elif density < 2.0:
        return 0.25 + 0.15 * (density - 0.5) / 1.5  # 0.25 to 0.4
    else:
        return 0.4  # max

def compute_D_local(R_val: float) -> float:
    return 0.4 - R_val

# ── World Initialization ─────────────────────────────────────────────────
def init_world(seed: int = 42):
    """Initialize world: obstacles, herbs, carcasses."""
    from .rng import reseed
    reseed(seed)
    rng = get_rng()
    
    # Clear grids
    for row in range(GRID_DIM):
        for col in range(GRID_DIM):
            animal_grid[row][col].clear()
            herb_grid[row][col].clear()
            obstacle_grid[row][col].clear()
            carcass_grid[row][col].clear()
    
    # Clear arrays
    obs_alive.fill(False)
    herb_alive.fill(False)
    carc_alive.fill(False)
    obs_I.fill(0); obs_R.fill(0); obs_D.fill(0)
    herb_ep.fill(0)
    carc_ep.fill(0); carc_age.fill(0)
    
    # Spawn initial obstacles (target ~3% density = 300 cells)
    n_obstacles = int(OBSTACLE_TARGET_DENSITY * GRID_DIM * GRID_DIM)
    n_obstacles = min(n_obstacles, MAX_OBSTACLES)
    
    obstacle_cells = set()
    while len(obstacle_cells) < n_obstacles:
        row = rng.integers(0, GRID_DIM)
        col = rng.integers(0, GRID_DIM)
        if (row, col) not in obstacle_cells:
            obstacle_cells.add((row, col))
    
    for i, (row, col) in enumerate(obstacle_cells):
        if i >= MAX_OBSTACLES:
            break
        obs_alive[i] = True
        obs_x[i] = (col + 0.5) * CELL_SIZE
        obs_y[i] = (row + 0.5) * CELL_SIZE
        obs_I[i] = 100.0
        obs_R[i] = 0.0
        obs_D[i] = 0.0
        obstacle_grid[row][col].add(f"OB{i:06d}")
        obstacle_positions[f"OB{i:06d}"] = (obs_x[i], obs_y[i])
    
    # Spawn initial herbs
    n_herbs = 20
    for i in range(n_herbs):
        row = rng.integers(0, GRID_DIM)
        col = rng.integers(0, GRID_DIM)
        herb_alive[i] = True
        herb_x[i] = (col + 0.5) * CELL_SIZE
        herb_y[i] = (row + 0.5) * CELL_SIZE
        herb_ep[i] = rng.uniform(10, 50)
        herb_grid[row][col].add(f"PL{i:06d}")
        herb_positions[f"PL{i:06d}"] = (herb_x[i], herb_y[i])
    
    print(f"✅ World initialized: {n_obstacles} obstacles, {n_herbs} herbs")

# ── Obstacle Update (Vectorized) ─────────────────────────────────────────
def update_obstacles_vectorized(turn: int):
    """Update all obstacles with local R/D computation."""
    alive_idx = np.where(obs_alive)[0]
    if len(alive_idx) == 0:
        return
    
    # Compute R/D for each obstacle based on its cell's local density
    R_vals = np.zeros(len(alive_idx), dtype=np.float32)
    D_vals = np.zeros(len(alive_idx), dtype=np.float32)
    
    for i, idx in enumerate(alive_idx):
        row = int(obs_y[idx] // CELL_SIZE)
        col = int(obs_x[idx] // CELL_SIZE)
        row = max(0, min(GRID_DIM - 1, row))
        col = max(0, min(GRID_DIM - 1, col))
        R_vals[i] = compute_R_local(row, col)
        D_vals[i] = compute_D_local(R_vals[i])
    
    # Update I/R/D
    obs_I[alive_idx] -= (R_vals + D_vals)
    obs_R[alive_idx] += R_vals
    obs_D[alive_idx] += D_vals
    
    # Determine actions
    total = obs_I[alive_idx] + obs_R[alive_idx] + obs_D[alive_idx]
    rolls = rng.random(len(alive_idx)) * total
    
    idle_mask = rolls < obs_I[alive_idx]
    replicate_mask = (rolls >= obs_I[alive_idx]) & (rolls < obs_I[alive_idx] + obs_R[alive_idx])
    destroy_mask = rolls >= (obs_I[alive_idx] + obs_R[alive_idx])
    
    # Handle replication
    if np.any(replicate_mask):
        rep_indices = alive_idx[replicate_mask]
        for idx in rep_indices:
            # Find empty neighboring cell
            row = int(obs_y[idx] // CELL_SIZE)
            col = int(obs_x[idx] // CELL_SIZE)
            placed = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < GRID_DIM and 0 <= nc < GRID_DIM:
                        if len(obstacle_grid[nr][nc]) == 0:
                            # Spawn new obstacle
                            free = np.where(~obs_alive)[0]
                            if len(free) > 0:
                                new_idx = free[0]
                                obs_alive[new_idx] = True
                                obs_x[new_idx] = (nc + 0.5) * CELL_SIZE
                                obs_y[new_idx] = (nr + 0.5) * CELL_SIZE
                                obs_I[new_idx] = 100.0
                                obs_R[new_idx] = 0.0
                                obs_D[new_idx] = 0.0
                                oid = f"OB{new_idx:06d}"
                                obstacle_grid[nr][nc].add(oid)
                                obstacle_positions[oid] = (obs_x[new_idx], obs_y[new_idx])
                                placed = True
                                break
                if placed:
                    break
            # Reset parent
            obs_I[idx] = 100.0
            obs_R[idx] = 0.0
            obs_D[idx] = 0.0
    
    # Handle destruction
    if np.any(destroy_mask):
        des_indices = alive_idx[destroy_mask]
        for idx in des_indices:
            row = int(obs_y[idx] // CELL_SIZE)
            col = int(obs_x[idx] // CELL_SIZE)
            oid = f"OB{idx:06d}"
            obstacle_grid[row][col].discard(oid)
            obstacle_positions.pop(oid, None)
            obs_alive[idx] = False

# ── Herb Update ──────────────────────────────────────────────────────────
def update_herbs_vectorized():
    """Herb EP recovery and replication with space check."""
    alive_idx = np.where(herb_alive)[0]
    if len(alive_idx) == 0:
        return
    
    # EP recovery
    herb_ep[alive_idx] += 1.0
    
    # Replication
    rng = get_rng()
    replicate_mask = (herb_ep[alive_idx] > 50) & (rng.random(len(alive_idx)) < 0.10)
    
    if np.any(replicate_mask):
        rep_indices = alive_idx[replicate_mask]
        for idx in rep_indices:
            row = int(herb_y[idx] // CELL_SIZE)
            col = int(herb_x[idx] // CELL_SIZE)
            # Check space in cell
            if len(herb_grid[row][col]) < 3:
                free = np.where(~herb_alive)[0]
                if len(free) > 0:
                    new_idx = free[0]
                    herb_alive[new_idx] = True
                    # Offset slightly
                    herb_x[new_idx] = herb_x[idx] + rng.uniform(-2, 2)
                    herb_y[new_idx] = herb_y[idx] + rng.uniform(-2, 2)
                    herb_x[new_idx] = max(0, min(WORLD_SIZE, herb_x[new_idx]))
                    herb_y[new_idx] = max(0, min(WORLD_SIZE, herb_y[new_idx]))
                    herb_ep[new_idx] = herb_ep[idx] * 0.5
                    herb_ep[idx] *= 0.5
                    hid = f"PL{new_idx:06d}"
                    new_row = int(herb_y[new_idx] // CELL_SIZE)
                    new_col = int(herb_x[new_idx] // CELL_SIZE)
                    herb_grid[new_row][new_col].add(hid)
                    herb_positions[hid] = (herb_x[new_idx], herb_y[new_idx])

# ── Carcass Update ───────────────────────────────────────────────────────
def update_carcasses_vectorized():
    """Carcass EP decay and removal."""
    alive_idx = np.where(carc_alive)[0]
    if len(alive_idx) == 0:
        return
    
    carc_ep[alive_idx] -= CARCASS_DECAY_RATE
    carc_age[alive_idx] += 1
    
    # Remove empty or old carcasses
    remove_mask = (carc_ep[alive_idx] <= 0) | (carc_age[alive_idx] > MAX_CARCASS_AGE)
    if np.any(remove_mask):
        rem_indices = alive_idx[remove_mask]
        for idx in rem_indices:
            row = int(carc_y[idx] // CELL_SIZE)
            col = int(carc_x[idx] // CELL_SIZE)
            cid = f"CA{idx:06d}"
            carcass_grid[row][col].discard(cid)
            carc_alive[idx] = False

# ── Spawn Carcass on Death ───────────────────────────────────────────────
def spawn_carcass(slot: int, max_energy_trait: float):
    """Create carcass at dead animal's position."""
    free = np.where(~carc_alive)[0]
    if len(free) == 0:
        return
    
    idx = free[0]
    carc_alive[idx] = True
    carc_x[idx] = x_pos[slot]
    carc_y[idx] = y_pos[slot]
    carc_ep[idx] = CARCASS_EP_FRACTION * max_energy_trait
    carc_age[idx] = 0
    
    row = int(carc_y[idx] // CELL_SIZE)
    col = int(carc_x[idx] // CELL_SIZE)
    cid = f"CA{idx:06d}"
    carcass_grid[row][col].add(cid)

# ── Grid Rebuild (for checkpoint resume) ─────────────────────────────────
def rebuild_grids():
    """Rebuild all spatial grids from current positions."""
    for row in range(GRID_DIM):
        for col in range(GRID_DIM):
            animal_grid[row][col].clear()
            herb_grid[row][col].clear()
            obstacle_grid[row][col].clear()
            carcass_grid[row][col].clear()
    
    obstacle_positions.clear()
    herb_positions.clear()
    
    # Obstacles
    for idx in np.where(obs_alive)[0]:
        row = int(obs_y[idx] // CELL_SIZE)
        col = int(obs_x[idx] // CELL_SIZE)
        oid = f"OB{idx:06d}"
        obstacle_grid[row][col].add(oid)
        obstacle_positions[oid] = (obs_x[idx], obs_y[idx])
    
    # Herbs
    for idx in np.where(herb_alive)[0]:
        row = int(herb_y[idx] // CELL_SIZE)
        col = int(herb_x[idx] // CELL_SIZE)
        hid = f"PL{idx:06d}"
        herb_grid[row][col].add(hid)
        herb_positions[hid] = (herb_x[idx], herb_y[idx])
    
    # Carcasses
    for idx in np.where(carc_alive)[0]:
        row = int(carc_y[idx] // CELL_SIZE)
        col = int(carc_x[idx] // CELL_SIZE)
        cid = f"CA{idx:06d}"
        carcass_grid[row][col].add(cid)