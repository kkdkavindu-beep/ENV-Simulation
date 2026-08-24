"""
Mutation and Reproduction — Vectorized, Persistent Mate-Seeking
"""
import numpy as np
import math
from .config import *
from .rng import get_rng
from .animal_model import (register_animal, release_slot, allocate_slot,
                          x_pos, y_pos, energy, alive, species,
                          genomes, genome_valid, generation,
                          animal_grid, grid_cell, animal_ids,
                          traits_shared, traits_carn, traits_herb,
                          M_F, infant_factor, mate_seek_timer)
from .genome_traits import (inherit_genome, populate_trait_cache,
                           load_weights_from_genome)

# ── Reproduction ─────────────────────────────────────────────────────────
def create_offspring(slot_A: int, slot_B: int) -> int | None:
    """
    Inherit genome, allocate slot, deduct reproduction cost, place near parent_A.
    Returns new slot index or None if no free slot.
    """
    # Final energy guard
    if energy[slot_A] < REPRODUCE_COST or energy[slot_B] < REPRODUCE_COST:
        return None
    
    slot = allocate_slot()
    if slot is None:
        return None  # population cap reached
    
    is_herb = bool(species[slot_A])  # both parents same species
    parent_genome_A = genomes[slot_A]
    parent_genome_B = genomes[slot_B]
    
    # Inherit genome
    child_genome = inherit_genome(parent_genome_A, parent_genome_B, is_herb)
    
    # Deduct reproduction cost from both parents
    energy[slot_A] -= REPRODUCE_COST
    energy[slot_B] -= REPRODUCE_COST
    
    # Compute child generation
    child_gen = int(max(generation[slot_A], generation[slot_B])) + 1
    
    # Place near parent A with small random offset
    rng = get_rng()
    off_x = float(x_pos[slot_A]) + rng.uniform(-5, 5)
    off_y = float(y_pos[slot_A]) + rng.uniform(-5, 5)
    off_x = max(0.0, min(WORLD_SIZE, off_x))
    off_y = max(0.0, min(WORLD_SIZE, off_y))
    
    # Populate trait cache from child genome
    populate_trait_cache(slot, child_genome, is_herb)
    
    # Load NN weights
    load_weights_from_genome(slot, child_genome, is_herb)
    
    # Register animal
    aid = register_animal(
        slot=slot,
        is_herb=is_herb,
        genome=child_genome,
        init_x=off_x,
        init_y=off_y,
        init_energy=10.0,
        init_age=0,
        trait_vals_shared=None,  # already populated
        trait_vals_carn=None,
        trait_vals_herb=None,
        genome_flat=child_genome,
        child_generation=child_gen
    )
    
    # Log birth
    log_birth(aid, animal_ids[slot_A], animal_ids[slot_B], off_x, off_y)
    
    return slot

def find_reproduction_pairs(reproduce_signals: np.ndarray) -> list[tuple[int, int]]:
    """
    Match animals in mate-seeking state within radius.
    Returns list of (slot_A, slot_B) pairs.
    """
    from .animal_model import x_pos, y_pos, alive, species
    
    pairs = []
    rng = get_rng()
    
    for is_herb in [True, False]:
        species_mask = species == is_herb
        # Animals seeking mates with sufficient energy
        seekers = np.where(
            species_mask & alive & (mate_seek_timer > 0) & (energy >= MIN_ENERGY_REPRODUCE)
        )[0]
        
        if len(seekers) < 2:
            continue
        
        paired = set()
        # Shuffle for fair pairing
        rng.shuffle(seekers)
        
        for i in range(len(seekers)):
            if seekers[i] in paired:
                continue
            ax, ay = float(x_pos[seekers[i]]), float(y_pos[seekers[i]])
            
            for j in range(i + 1, len(seekers)):
                if seekers[j] in paired:
                    continue
                bx, by = float(x_pos[seekers[j]]), float(y_pos[seekers[j]])
                dist_sq = (ax - bx) ** 2 + (ay - by) ** 2
                
                if dist_sq <= REPRODUCE_RADIUS ** 2:
                    pairs.append((int(seekers[i]), int(seekers[j])))
                    paired.add(seekers[i])
                    paired.add(seekers[j])
                    break
    
    return pairs

# ── Mate Seeking ─────────────────────────────────────────────────────────
def update_mate_seek(herb_idx: np.ndarray, carn_idx: np.ndarray,
                     decoded_h: dict, decoded_c: dict):
    """Update mate_seek_timer based on NN output."""
    # Herbivores
    for i, slot in enumerate(herb_idx):
        if decoded_h["mate_seek"][i]:
            mate_seek_timer[slot] = MATE_SEEK_DURATION
        elif mate_seek_timer[slot] > 0:
            pass  # already seeking, let pairing handle it
    
    # Carnivores
    for i, slot in enumerate(carn_idx):
        if decoded_c["mate_seek"][i]:
            mate_seek_timer[slot] = MATE_SEEK_DURATION
        elif mate_seek_timer[slot] > 0:
            pass

# ── Logging ──────────────────────────────────────────────────────────────
def log_birth(child_id: str, parent_a_id: str, parent_b_id: str, x: float, y: float):
    """Log birth event — called from create_offspring."""
    # This will be connected to logging module
    pass