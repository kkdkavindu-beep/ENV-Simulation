"""
Animal Data Model — Pure NumPy SoA Layout
Zero Python objects in hot path.
"""
import numpy as np
from .config import *
from .rng import get_rng

# ── SoA Arrays ───────────────────────────────────────────────────────────
alive          = np.zeros(MAX_ANIMALS, dtype=np.bool_)
species        = np.zeros(MAX_ANIMALS, dtype=np.bool_)   # True=herb, False=carn
x_pos          = np.zeros(MAX_ANIMALS, dtype=np.float32)
y_pos          = np.zeros(MAX_ANIMALS, dtype=np.float32)
energy         = np.zeros(MAX_ANIMALS, dtype=np.float32)
health         = np.zeros(MAX_ANIMALS, dtype=np.float32)
fatigue        = np.zeros(MAX_ANIMALS, dtype=np.float32)
age            = np.zeros(MAX_ANIMALS, dtype=np.int32)

# Traits: shared (9)
traits_shared  = np.zeros((MAX_ANIMALS, 9), dtype=np.float32)
# Carnivore-only (3)
traits_carn    = np.zeros((MAX_ANIMALS, 3), dtype=np.float32)
# Herbivore-only (3)
traits_herb    = np.zeros((MAX_ANIMALS, 3), dtype=np.float32)

# Cognition
memory         = np.zeros((MAX_ANIMALS, 16), dtype=np.uint8)

# Active stealth boost for current tick (herbivores only)
active_stealth_boost = np.zeros(MAX_ANIMALS, dtype=np.float32)

# Fatigue multiplier M(F) — computed once per tick
M_F            = np.ones(MAX_ANIMALS, dtype=np.float32)

# Infant factor — computed once per tick
infant_factor  = np.ones(MAX_ANIMALS, dtype=np.float32)

# Mate-seeking state (persistent)
mate_seek_timer = np.zeros(MAX_ANIMALS, dtype=np.int16)

# Last detection results (Python list - ragged)
last_scan: list[list] = [[] for _ in range(MAX_ANIMALS)]

# Genome storage (compressed binary flat arrays)
genomes        = np.zeros((MAX_ANIMALS, GENOME_SIZE_REDUCED), dtype=np.float32)
genome_valid   = np.zeros(MAX_ANIMALS, dtype=np.bool_)

# Generation counter
generation     = np.zeros(MAX_ANIMALS, dtype=np.int32)

# Neural Network Weight Tensors (pre-reshaped, per species)
W1_h = np.zeros((MAX_ANIMALS, HIDDEN_SIZE, INPUT_SIZE), dtype=np.float32)
W2_h = np.zeros((MAX_ANIMALS, HIDDEN_SIZE, HIDDEN_SIZE), dtype=np.float32)
W3_h = np.zeros((MAX_ANIMALS, OUTPUT_SIZE, HIDDEN_SIZE), dtype=np.float32)
b1_h = np.zeros((MAX_ANIMALS, HIDDEN_SIZE), dtype=np.float32)
b2_h = np.zeros((MAX_ANIMALS, HIDDEN_SIZE), dtype=np.float32)
b3_h = np.zeros((MAX_ANIMALS, OUTPUT_SIZE), dtype=np.float32)

W1_c = np.zeros((MAX_ANIMALS, HIDDEN_SIZE, INPUT_SIZE), dtype=np.float32)
W2_c = np.zeros((MAX_ANIMALS, HIDDEN_SIZE, HIDDEN_SIZE), dtype=np.float32)
W3_c = np.zeros((MAX_ANIMALS, OUTPUT_SIZE, HIDDEN_SIZE), dtype=np.float32)
b1_c = np.zeros((MAX_ANIMALS, HIDDEN_SIZE), dtype=np.float32)
b2_c = np.zeros((MAX_ANIMALS, HIDDEN_SIZE), dtype=np.float32)
b3_c = np.zeros((MAX_ANIMALS, OUTPUT_SIZE), dtype=np.float32)

# ── ID Mapping (Python side only) ────────────────────────────────────────
animal_ids: list[str | None] = [None] * MAX_ANIMALS
id_to_slot: dict[str, int] = {}

next_herb_id = 0
next_carn_id = 0

# ── Simulation State ─────────────────────────────────────────────────────
# Turn counter (set by tick_loop, read by logging)
turn = 0

# Last decoded NN outputs per animal (for next tick's detection params)
last_decoded: list[dict | None] = [None] * MAX_ANIMALS

# Sound inbox: (N, 4, 3) float32 — [strength, type, sender_id], sorted strongest-first
sound_inbox = np.zeros((MAX_ANIMALS, 4, 3), dtype=np.float32)

# ── Spatial Grids (set by world module) ──────────────────────────────────
animal_grid   = [[set() for _ in range(GRID_DIM)] for _ in range(GRID_DIM)]
herb_grid     = [[set() for _ in range(GRID_DIM)] for _ in range(GRID_DIM)]
obstacle_grid = [[set() for _ in range(GRID_DIM)] for _ in range(GRID_DIM)]
carcass_grid  = [[set() for _ in range(GRID_DIM)] for _ in range(GRID_DIM)]

# Herb positions (for detection)
herb_positions: dict[str, tuple] = {}

# Obstacle positions
obstacle_positions: dict[str, tuple] = {}

# Carcass storage
carc_x = np.zeros(MAX_CARCASSES, dtype=np.float32)
carc_y = np.zeros(MAX_CARCASSES, dtype=np.float32)
carc_ep = np.zeros(MAX_CARCASSES, dtype=np.float32)
carc_age = np.zeros(MAX_CARCASSES, dtype=np.int32)
carc_alive = np.zeros(MAX_CARCASSES, dtype=np.bool_)

# Herb storage
herb_x = np.zeros(MAX_HERBS, dtype=np.float32)
herb_y = np.zeros(MAX_HERBS, dtype=np.float32)
herb_ep = np.zeros(MAX_HERBS, dtype=np.float32)
herb_alive = np.zeros(MAX_HERBS, dtype=np.bool_)

# Obstacle storage
obs_I = np.zeros(MAX_OBSTACLES, dtype=np.float32)
obs_R = np.zeros(MAX_OBSTACLES, dtype=np.float32)
obs_D = np.zeros(MAX_OBSTACLES, dtype=np.float32)
obs_x = np.zeros(MAX_OBSTACLES, dtype=np.float32)
obs_y = np.zeros(MAX_OBSTACLES, dtype=np.float32)
obs_alive = np.zeros(MAX_OBSTACLES, dtype=np.bool_)

# ── Grid Helpers ─────────────────────────────────────────────────────────
def grid_cell(x: float, y: float) -> tuple[int, int]:
    """Convert world coordinates to grid cell (row, col)."""
    col = int(x // CELL_SIZE)
    row = int(y // CELL_SIZE)
    col = max(0, min(GRID_DIM - 1, col))
    row = max(0, min(GRID_DIM - 1, row))
    return row, col

# ── Slot Management ──────────────────────────────────────────────────────
def allocate_slot() -> int | None:
    """Find first unused slot."""
    free = np.where(~alive)[0]
    return int(free[0]) if len(free) > 0 else None

def reset_arrays():
    """Reset all arrays for new run."""
    global next_herb_id, next_carn_id, animal_ids, id_to_slot, last_scan, last_decoded, sound_inbox, turn
    
    turn = 0
    alive.fill(False)
    species.fill(False)
    x_pos.fill(0)
    y_pos.fill(0)
    energy.fill(0)
    health.fill(0)
    fatigue.fill(0)
    age.fill(0)
    memory.fill(0)
    active_stealth_boost.fill(0)
    M_F.fill(1.0)
    infant_factor.fill(1.0)
    mate_seek_timer.fill(0)
    traits_shared.fill(0)
    traits_carn.fill(0)
    traits_herb.fill(0)
    genomes.fill(0)
    genome_valid.fill(False)
    generation.fill(0)
    
    W1_h.fill(0); W2_h.fill(0); W3_h.fill(0)
    b1_h.fill(0); b2_h.fill(0); b3_h.fill(0)
    W1_c.fill(0); W2_c.fill(0); W3_c.fill(0)
    b1_c.fill(0); b2_c.fill(0); b3_c.fill(0)
    
    animal_ids = [None] * MAX_ANIMALS
    id_to_slot.clear()
    last_scan = [[] for _ in range(MAX_ANIMALS)]
    last_decoded = [None] * MAX_ANIMALS
    sound_inbox.fill(0.0)
    
    next_herb_id = 0
    next_carn_id = 0

def register_animal(slot: int, is_herb: bool,
                    init_x: float, init_y: float,
                    init_energy: float, init_age: int,
                    trait_vals_shared: np.ndarray,
                    trait_vals_carn: np.ndarray | None,
                    trait_vals_herb: np.ndarray | None,
                    genome_flat: np.ndarray,
                    child_generation: int) -> str:
    """Set all SoA fields for a new animal and return its ID."""
    global next_herb_id, next_carn_id
    
    alive[slot]      = True
    species[slot]    = is_herb
    x_pos[slot]      = init_x
    y_pos[slot]      = init_y
    energy[slot]     = init_energy
    fatigue[slot]    = 0.0
    age[slot]        = init_age
    memory[slot]     = 0
    last_scan[slot]  = []
    active_stealth_boost[slot] = 0.0
    mate_seek_timer[slot] = 0

    # Traits
    traits_shared[slot] = trait_vals_shared
    if is_herb:
        traits_herb[slot] = trait_vals_herb
        traits_carn[slot] = 0
    else:
        traits_carn[slot] = trait_vals_carn
        traits_herb[slot] = 0

    health[slot] = trait_vals_shared[T_HEALTH]

    # Genome
    genomes[slot] = genome_flat
    genome_valid[slot] = True
    generation[slot] = child_generation

    # Assign ID
    if is_herb:
        aid = f"HB{next_herb_id:06d}"
        next_herb_id += 1
    else:
        aid = f"CN{next_carn_id:06d}"
        next_carn_id += 1

    animal_ids[slot] = aid
    id_to_slot[aid] = slot

    # Add to spatial grid
    r, c = grid_cell(init_x, init_y)
    animal_grid[r][c].add(aid)

    return aid

def release_slot(slot: int) -> None:
    """Remove animal from all structures. Called in Phase 8."""
    aid = animal_ids[slot]
    if aid:
        r, c = grid_cell(float(x_pos[slot]), float(y_pos[slot]))
        animal_grid[r][c].discard(aid)
        id_to_slot.pop(aid, None)
    alive[slot]           = False
    animal_ids[slot]      = None
    genome_valid[slot]    = False
    generation[slot]      = 0
    mate_seek_timer[slot] = 0
    last_scan[slot]       = []

# ── Trait Access Helpers ─────────────────────────────────────────────────
def get_effective_speed(slots: np.ndarray) -> np.ndarray:
    base = traits_shared[slots, T_SPEED]
    return base * infant_factor[slots] * M_F[slots]

def get_effective_endurance(slots: np.ndarray) -> np.ndarray:
    return traits_shared[slots, T_ENDURANCE]

def get_effective_attack(slots: np.ndarray) -> np.ndarray:
    base = traits_carn[slots, TC_ATTACK]
    return base * infant_factor[slots] * M_F[slots]

def get_effective_camo(slots: np.ndarray) -> np.ndarray:
    base = traits_herb[slots, TH_CAMO_EFF]
    return base * infant_factor[slots]

def get_effective_stealth_power(slots: np.ndarray) -> np.ndarray:
    base = traits_herb[slots, TH_STEALTH_PWR]
    return base * infant_factor[slots] * M_F[slots]

def get_effective_bite_eff(slots: np.ndarray) -> np.ndarray:
    base = traits_carn[slots, TC_BITE_EFF]
    return base * infant_factor[slots]

def get_effective_det_eff(slots: np.ndarray) -> np.ndarray:
    base = traits_shared[slots, T_DET_EFF]
    return base * infant_factor[slots]

def get_effective_sound_power(slots: np.ndarray) -> np.ndarray:
    base = traits_shared[slots, T_SND_PWR]
    return base * infant_factor[slots] * M_F[slots]

# ── Death Check ──────────────────────────────────────────────────────────
def get_dead_mask() -> np.ndarray:
    """Vectorized death mask."""
    lifespan_arr = traits_shared[:, T_LIFESPAN].astype(np.int32)
    return alive & (
        (energy <= 0) |
        (health <= 0) |
        (age >= lifespan_arr)
    )

# ── Population Counts ────────────────────────────────────────────────────
def count_herbivores() -> int:
    return int(np.sum(alive & species))

def count_carnivores() -> int:
    return int(np.sum(alive & ~species))