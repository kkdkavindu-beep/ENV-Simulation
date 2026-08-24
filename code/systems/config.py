"""
Configuration constants — single source of truth for all magic numbers.
"""
import numpy as np
from datetime import datetime

# ── Global ──────────────────────────────────────────────────────────────
MAX_ANIMALS   = 2000
MAX_HERBS     = 500
MAX_OBSTACLES = 600
MAX_CARCASSES = 200

WORLD_SIZE    = 1000.0
CELL_SIZE     = 10.0
GRID_DIM      = int(WORLD_SIZE / CELL_SIZE)  # 100

# ── Time ────────────────────────────────────────────────────────────────
TURN_MS       = 100
TURNS_PER_SEC = 10

# ── Neural Network ──────────────────────────────────────────────────────
USE_REDUCED_ARCH   = True
INPUT_SIZE         = 32
HIDDEN_SIZE        = 16
OUTPUT_SIZE        = 16
NUM_LAYERS         = 2

# Full architecture (if not reduced)
FULL_INPUT_SIZE    = 44
FULL_OUTPUT_SIZE   = 29

# ── Detection ───────────────────────────────────────────────────────────
R_MAX              = 200.0
NUM_DIR_BINS       = 8
ANGLE_LEVELS       = 3
ANGLE_MAP          = np.array([np.pi/6, np.pi/3, np.pi/2], dtype=np.float32)
DETECTION_COST_FACTOR = 0.005
DETECTION_FATIGUE_LARGE = 0.5
DETECTION_FATIGUE_SMALL = 0.1

# ── Sound ───────────────────────────────────────────────────────────────
ATTENUATION_K      = 0.021
SOUND_INBOX_SIZE   = 4
CARCASS_SCENT_TYPE = 15

# ── Fatigue ─────────────────────────────────────────────────────────────
FATIGUE_RECOVERY   = 0.2
FATIGUE_THRESHOLD_FRAC = 0.3  # 30% of endurance
FATIGUE_EXPONENT   = 1.5
FATIGUE_MIN_MULT   = 0.01
MOVE_FATIGUE_FACTOR = 0.15
ATTACK_FATIGUE     = 5.0

# ── Energy ──────────────────────────────────────────────────────────────
LIVING_COST_FACTOR     = 0.001
MOVE_COST_FACTOR       = 0.03
ATTACK_COST_FACTOR     = 0.2
SOUND_COST_FACTOR      = 0.001
STEALTH_COST_FACTOR    = 0.01
TRANSFER_FRACTION      = 0.10
TRANSFER_CAP           = 10.0
BASE_FEED_RATE         = 5.0
CARNIVORE_HERB_EFF     = 0.1

# ── Reproduction ────────────────────────────────────────────────────────
REPRODUCE_RADIUS       = 15.0
MIN_ENERGY_REPRODUCE   = 60.0
REPRODUCE_COST         = 30.0
MATE_SEEK_DURATION     = 100  # ticks

# ── Infant ──────────────────────────────────────────────────────────────
INFANT_LIFESPAN_FRAC   = 0.10

# ── Memory ──────────────────────────────────────────────────────────────
MEMORY_CELLS           = 16
MEMORY_READ_CELLS      = 4
MEMORY_BLEND_OLD       = 7
MEMORY_BLEND_NEW       = 3
MEMORY_BLEND_DIV       = 10

# ── Carcass ─────────────────────────────────────────────────────────────
CARCASS_EP_FRACTION    = 0.5
CARCASS_DECAY_RATE     = 2.0
MAX_CARCASS_AGE        = 50

# ── Obstacle ────────────────────────────────────────────────────────────
OBSTACLE_TARGET_DENSITY = 0.03  # 3% of cells
OBSTACLE_R_FUNC = None  # set in world module

# ── Logging ─────────────────────────────────────────────────────────────
PUSH_INTERVAL        = 10
CHECKPOINT_INTERVAL  = 500
LOG_SAMPLE_RATE      = 10
LOG_MODE             = "event_only"  # "full", "sampled", "event_only"

# ── Simulation Speed ────────────────────────────────────────────────────
TARGET_SIM_SPEED     = 50  # turns/sec

# ── RNG ─────────────────────────────────────────────────────────────────
DEFAULT_SEED         = 42

# ── Trait Indices ───────────────────────────────────────────────────────
# Shared traits (traits_shared columns)
T_HEALTH      = 0
T_MAX_ENERGY  = 1
T_LIFESPAN    = 2
T_SPEED       = 3
T_ENDURANCE   = 4
T_METABOLISM  = 5
T_DET_EFF     = 6
T_SND_SENS    = 7
T_SND_PWR     = 8

# Carnivore traits (traits_carn columns)
TC_ATTACK     = 0
TC_STEALTH    = 1
TC_BITE_EFF   = 2

# Herbivore traits (traits_herb columns)
TH_CAMO_EFF   = 0
TH_DEFENCE    = 1
TH_STEALTH_PWR = 2

# ── Genome Layout ───────────────────────────────────────────────────────
TRAIT_ALLELE_SIZE = 540      # 15 traits × 6 genes × 2 alleles × 3 fields
BRAIN_WEIGHTS_FULL = 1485
BRAIN_WEIGHTS_REDUCED = 800
GENOME_SIZE_FULL = TRAIT_ALLELE_SIZE + BRAIN_WEIGHTS_FULL
GENOME_SIZE_REDUCED = TRAIT_ALLELE_SIZE + BRAIN_WEIGHTS_REDUCED

# Mutation rates
TRUE_VALUE_MUTATION_RATE = 0.01
TRUE_VALUE_MUTATION_SIZE = 0.05
DOMINANCE_MUTATION_RATE = 0.05
DOMINANCE_MUTATION_STEP = 1.0
BRAIN_MUTATION_RATE = 0.20
BRAIN_MUTATION_SIZE = 0.05

# Dominance encoding
DOMINANCE_MIN = -8
DOMINANCE_MAX = 7

# ── Species Defaults ────────────────────────────────────────────────────
SPECIES_DEFAULTS = {
    "shared": {
        "health": 84, "max_energy": 540, "lifespan": 550, "speed": 24,
        "endurance": 90, "metabolism": 92, "detection_efficiency": 92,
        "sound_sensitivity": 21, "sound_power": 22,
    },
    "carnivore": {
        "attack_power": 12, "stealth": 12, "bite_efficiency": 60,
    },
    "herbivore": {
        "camouflage_efficiency": 90, "defence": 4, "active_stealth_power": 20,
    },
}

# ── Web UI ──────────────────────────────────────────────────────────────
WS_PUSH_INTERVAL = 10
API_PORT = 8080
# ── Drive / Run Paths ───────────────────────────────────────────
DRIVE_BASE = "/content/drive/MyDrive/envsim"
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H%M%S")
