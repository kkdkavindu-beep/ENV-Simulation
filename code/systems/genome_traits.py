"""
Genome and Traits — Flat Binary Layout, Vectorized Operations
"""
import numpy as np
from config import *
from rng import get_rng

# Trait name list for JSON serialization
TRAIT_NAMES = [
    "health", "max_energy", "lifespan", "speed", "endurance",
    "metabolism", "detection_efficiency", "sound_sensitivity", "sound_power",
    "attack_power", "stealth", "bite_efficiency",
    "camouflage_efficiency", "defence", "active_stealth_power"
]

# ── Dominance Resolution (Vectorized) ────────────────────────────────────
def resolve_dominance_vectorized(genome_flat: np.ndarray) -> np.ndarray:
    """
    Compute effective trait values from genome alleles.
    Returns array of 15 trait values.
    """
    traits = np.zeros(15, dtype=np.float32)
    rng = get_rng()
    
    for trait_idx in range(15):
        trait_base = trait_idx * 36  # 6 genes × 2 alleles × 3 fields
        total = 0.0
        for gene_idx in range(6):
            gene_base = trait_base + gene_idx * 6
            # Allele 0
            tv0 = genome_flat[gene_base + 0]
            mv0 = genome_flat[gene_base + 1]
            dv0 = genome_flat[gene_base + 2]
            # Allele 1
            tv1 = genome_flat[gene_base + 3]
            mv1 = genome_flat[gene_base + 4]
            dv1 = genome_flat[gene_base + 5]
            
            # Dominance: higher wins; tie = random (deterministic from RNG)
            if dv0 > dv1:
                total += tv0 + mv0
            elif dv1 > dv0:
                total += tv1 + mv1
            else:
                total += (tv0 + mv0) if rng.random() < 0.5 else (tv1 + mv1)
        traits[trait_idx] = total
    
    return traits

# ── Progenitor Genome ────────────────────────────────────────────────────
def create_progenitor_genome(is_herb: bool) -> np.ndarray:
    """Return flat float32 array of genome size."""
    rng = get_rng()
    n_weights = BRAIN_WEIGHTS_REDUCED if USE_REDUCED_ARCH else BRAIN_WEIGHTS_FULL
    genome = np.zeros(TRAIT_ALLELE_SIZE + n_weights, dtype=np.float32)
    
    defaults = {**SPECIES_DEFAULTS["shared"]}
    if is_herb:
        defaults.update(SPECIES_DEFAULTS["herbivore"])
    else:
        defaults.update(SPECIES_DEFAULTS["carnivore"])
    
    trait_names = list(defaults.keys())
    for trait_idx, trait_name in enumerate(trait_names):
        total_default = defaults[trait_name]
        per_gene = total_default / 6.0
        
        trait_base = trait_idx * 36
        for gene_idx in range(6):
            gene_base = trait_base + gene_idx * 6
            for allele_idx in range(2):
                allele_base = gene_base + allele_idx * 3
                jitter = per_gene * rng.uniform(-0.20, 0.20)
                genome[allele_base + 0] = per_gene + jitter  # true_value
                genome[allele_base + 1] = 0.0                 # mutation_value
                genome[allele_base + 2] = rng.integers(DOMINANCE_MIN, DOMINANCE_MAX + 1)  # dominance
    
    # Brain weights
    genome[TRAIT_ALLELE_SIZE:] = rng.normal(0.0, 0.1, n_weights).astype(np.float32)
    
    return genome

# ── Trait Cache Population ───────────────────────────────────────────────
def populate_trait_cache(slot: int, genome_flat: np.ndarray, is_herb: bool):
    """Compute traits from genome and store in SoA arrays."""
    traits = resolve_dominance_vectorized(genome_flat)
    
    # Shared traits (9)
    from animal_model import traits_shared, traits_carn, traits_herb
    traits_shared[slot] = traits[0:9]
    
    if is_herb:
        # Herbivore traits at indices 12,13,14 -> 0,1,2
        traits_herb[slot, 0] = traits[12]
        traits_herb[slot, 1] = traits[13]
        traits_herb[slot, 2] = traits[14]
        traits_carn[slot] = 0
    else:
        # Carnivore traits at indices 9,10,11 -> 0,1,2
        traits_carn[slot, 0] = traits[9]
        traits_carn[slot, 1] = traits[10]
        traits_carn[slot, 2] = traits[11]
        traits_herb[slot] = 0

# ── Inheritance (Vectorized) ─────────────────────────────────────────────
def inherit_trait_alleles(genome_A: np.ndarray, genome_B: np.ndarray) -> np.ndarray:
    """Vectorized inheritance for trait alleles (first TRAIT_ALLELE_SIZE values)."""
    rng = get_rng()
    child = np.zeros(TRAIT_ALLELE_SIZE, dtype=np.float32)
    
    # 50/50 parent choice
    mask = rng.random(TRAIT_ALLELE_SIZE) < 0.5
    child[mask] = genome_A[mask]
    child[~mask] = genome_B[~mask]
    
    # Mutate true_value (1% chance, ±5%)
    true_value_mask = (np.arange(TRAIT_ALLELE_SIZE) % 3 == 0)
    mut_mask = true_value_mask & (rng.random(TRAIT_ALLELE_SIZE) < TRUE_VALUE_MUTATION_RATE)
    child[mut_mask] *= 1.0 + rng.uniform(-TRUE_VALUE_MUTATION_SIZE, TRUE_VALUE_MUTATION_SIZE, mut_mask.sum())
    
    # Mutate dominance_value (5% chance, ±1 clamped)
    dom_mask = (np.arange(TRAIT_ALLELE_SIZE) % 3 == 2)
    mut_mask = dom_mask & (rng.random(TRAIT_ALLELE_SIZE) < DOMINANCE_MUTATION_RATE)
    child[mut_mask] += rng.choice([-1.0, 1.0], mut_mask.sum())
    np.clip(child[dom_mask], DOMINANCE_MIN, DOMINANCE_MAX, out=child[dom_mask])
    
    return child

def inherit_brain_weights(genome_A: np.ndarray, genome_B: np.ndarray) -> np.ndarray:
    """Vectorized brain weight inheritance with mutation."""
    rng = get_rng()
    n = BRAIN_WEIGHTS_REDUCED if USE_REDUCED_ARCH else BRAIN_WEIGHTS_FULL
    offset = TRAIT_ALLELE_SIZE
    
    wa = genome_A[offset:offset+n]
    wb = genome_B[offset:offset+n]
    
    # 50/50 blend
    mask = rng.random(n) < 0.5
    child = np.where(mask, wa, wb)
    
    # 20% mutation, ±5% of magnitude
    mut_mask = rng.random(n) < BRAIN_MUTATION_RATE
    signs = rng.choice([-1.0, 1.0], mut_mask.sum())
    deltas = np.abs(child[mut_mask]) * BRAIN_MUTATION_SIZE * signs
    child[mut_mask] += deltas
    
    return child

def inherit_genome(genome_A: np.ndarray, genome_B: np.ndarray, is_herb: bool) -> np.ndarray:
    """Full genome inheritance."""
    n_weights = BRAIN_WEIGHTS_REDUCED if USE_REDUCED_ARCH else BRAIN_WEIGHTS_FULL
    child = np.zeros(TRAIT_ALLELE_SIZE + n_weights, dtype=np.float32)
    
    child[:TRAIT_ALLELE_SIZE] = inherit_trait_alleles(genome_A, genome_B)
    child[TRAIT_ALLELE_SIZE:] = inherit_brain_weights(genome_A, genome_B)
    
    return child

# ── Weight Loading ───────────────────────────────────────────────────────
def load_weights_from_genome(slot: int, genome_flat: np.ndarray, is_herb: bool):
    """Reshape genome brain weights into SoA tensors. Called once at creation."""
    from animal_model import (W1_h, b1_h, W2_h, b2_h, W3_h, b3_h,
                               W1_c, b1_c, W2_c, b2_c, W3_c, b3_c)
    
    n_weights = BRAIN_WEIGHTS_REDUCED if USE_REDUCED_ARCH else BRAIN_WEIGHTS_FULL
    w = genome_flat[TRAIT_ALLELE_SIZE:TRAIT_ALLELE_SIZE + n_weights]
    
    if USE_REDUCED_ARCH:
        # Reduced: W1(16,32)=512, b1(16), W2(16,16)=256, b2(16), W3(16,16)=256, b3(16) = ~800
        if is_herb:
            W1_h[slot] = w[0:512].reshape(HIDDEN_SIZE, INPUT_SIZE)
            b1_h[slot] = w[512:528]
            W2_h[slot] = w[528:784].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
            b2_h[slot] = w[784:800]
            W3_h[slot] = w[800:1056].reshape(OUTPUT_SIZE, HIDDEN_SIZE)
            b3_h[slot] = w[1056:1072]
        else:
            W1_c[slot] = w[0:512].reshape(HIDDEN_SIZE, INPUT_SIZE)
            b1_c[slot] = w[512:528]
            W2_c[slot] = w[528:784].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
            b2_c[slot] = w[784:800]
            W3_c[slot] = w[800:1056].reshape(OUTPUT_SIZE, HIDDEN_SIZE)
            b3_c[slot] = w[1056:1072]
    else:
        # Full architecture
        if is_herb:
            W1_h[slot] = w[0:704].reshape(16, 44)
            b1_h[slot] = w[704:720]
            W2_h[slot] = w[720:976].reshape(16, 16)
            b2_h[slot] = w[976:992]
            W3_h[slot] = w[992:1456].reshape(29, 16)
            b3_h[slot] = w[1456:1485]
        else:
            W1_c[slot] = w[0:704].reshape(16, 44)
            b1_c[slot] = w[704:720]
            W2_c[slot] = w[720:976].reshape(16, 16)
            b2_c[slot] = w[976:992]
            W3_c[slot] = w[992:1456].reshape(29, 16)
            b3_c[slot] = w[1456:1485]

# ── JSON Serialization (for logging) ─────────────────────────────────────
def genome_to_json(genome_flat: np.ndarray) -> dict:
    """Convert flat genome to nested dict for logging."""
    result = {}
    
    for trait_idx, name in enumerate(TRAIT_NAMES):
        result[name] = {}
        trait_base = trait_idx * 36
        for gene_idx in range(6):
            result[name][gene_idx] = {}
            gene_base = trait_base + gene_idx * 6
            for allele_idx in range(2):
                allele_base = gene_base + allele_idx * 3
                result[name][gene_idx][allele_idx] = {
                    "true_value": float(genome_flat[allele_base + 0]),
                    "mutation_value": float(genome_flat[allele_base + 1]),
                    "dominance_value": int(genome_flat[allele_base + 2]),
                }
    
    # Brain weights (omit for brevity in logs, or include hash)
    n_weights = BRAIN_WEIGHTS_REDUCED if USE_REDUCED_ARCH else BRAIN_WEIGHTS_FULL
    result["brain_weights_hash"] = hash(genome_flat[TRAIT_ALLELE_SIZE:TRAIT_ALLELE_SIZE+n_weights].tobytes())
    
    return result

# ── Infant Factor ────────────────────────────────────────────────────────
def compute_infant_factor_vectorized(alive_idx: np.ndarray):
    """Compute infant factor for all alive animals."""
    from animal_model import age, traits_shared, infant_factor
    
    lifespan = traits_shared[alive_idx, T_LIFESPAN].astype(np.float32)
    infant_thresh = INFANT_LIFESPAN_FRAC * lifespan
    is_infant = age[alive_idx] < infant_thresh
    
    infant_factor[alive_idx] = 1.0
    if np.any(is_infant):
        infant_idx = alive_idx[is_infant]
        infant_factor[infant_idx] = (
            age[infant_idx].astype(np.float32) / infant_thresh[is_infant]
        )
    np.clip(infant_factor, 0.0, 1.0, out=infant_factor)