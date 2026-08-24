"""
Energy System — Unified Phase 5 Cost Application
All energy costs applied in single vectorized pass.
"""
import numpy as np
from config import *
from animal_model import (energy, alive, traits_shared, traits_carn, traits_herb,
                          M_F, infant_factor, x_pos, y_pos, fatigue, health,
                          active_stealth_boost)

# ── Cost Computation (Vectorized) ────────────────────────────────────────
def apply_living_cost(alive_idx: np.ndarray):
    """Living cost: 0.001 * metabolism"""
    energy[alive_idx] -= LIVING_COST_FACTOR * traits_shared[alive_idx, T_METABOLISM]

def apply_movement_cost(move_slots: np.ndarray, dx: np.ndarray, dy: np.ndarray):
    """Movement cost: 0.03 * (dx² + dy²)"""
    if len(move_slots) == 0:
        return
    move_cost = MOVE_COST_FACTOR * (dx * dx + dy * dy)
    energy[move_slots] -= move_cost
    # Movement fatigue
    from animal_model import fatigue
    fatigue[move_slots] += MOVE_FATIGUE_FACTOR * (np.abs(dx) + np.abs(dy))

def apply_detection_cost(detect_slots: np.ndarray, R_eff: np.ndarray, alpha: np.ndarray):
    """Detection cost: 0.005 * R² * (α/π)"""
    if len(detect_slots) == 0:
        return
    det_cost = DETECTION_COST_FACTOR * R_eff * R_eff * (alpha / np.pi)
    energy[detect_slots] -= det_cost
    # Detection fatigue
    from animal_model import fatigue
    fatigue[detect_slots] += np.where(R_eff > 50, DETECTION_FATIGUE_LARGE, DETECTION_FATIGUE_SMALL)

def apply_attack_cost(attacker_slots: np.ndarray, eff_attack: np.ndarray):
    """Attack cost: 0.2 * effective_attack"""
    if len(attacker_slots) == 0:
        return
    atk_cost = ATTACK_COST_FACTOR * eff_attack
    energy[attacker_slots] -= atk_cost
    # Attack fatigue
    from animal_model import fatigue
    fatigue[attacker_slots] += ATTACK_FATIGUE

def apply_sound_cost(emit_slots: np.ndarray, S_emit: np.ndarray):
    """Sound cost: 0.001 * S_emit"""
    if len(emit_slots) == 0:
        return
    snd_cost = SOUND_COST_FACTOR * S_emit
    energy[emit_slots] -= snd_cost

def apply_stealth_cost(stealth_slots: np.ndarray, boost: np.ndarray):
    """Active stealth cost: 0.01 * boost"""
    if len(stealth_slots) == 0:
        return
    stealth_cost = STEALTH_COST_FACTOR * boost
    energy[stealth_slots] -= stealth_cost

def apply_transfer_cost(actor_slots: np.ndarray, transfer_amt: np.ndarray):
    """Energy transfer cost: min(energy * 0.1, 10)"""
    if len(actor_slots) == 0:
        return
    energy[actor_slots] -= transfer_amt

# ── Unified Cost Application ─────────────────────────────────────────────
def apply_all_costs(move_queue, detect_fired, damage_queue,
                    sound_emit_queue, stealth_queue, transfer_queue,
                    fatigue_delta):
    """
    Single vectorized pass for ALL energy costs.
    Called in Phase 5.
    """
    # Movement
    if move_queue:
        slots = np.array([m[0] for m in move_queue], dtype=np.int32)
        dx = np.array([m[3] for m in move_queue], dtype=np.float32)
        dy = np.array([m[4] for m in move_queue], dtype=np.float32)
        apply_movement_cost(slots, dx, dy)
        fatigue_delta[slots] += MOVE_FATIGUE_FACTOR * (np.abs(dx) + np.abs(dy))
    
    # Detection
    if detect_fired:
        slots = np.array([d[0] for d in detect_fired], dtype=np.int32)
        R_eff = np.array([d[1] for d in detect_fired], dtype=np.float32)
        alpha = np.array([d[2] for d in detect_fired], dtype=np.float32)
        apply_detection_cost(slots, R_eff, alpha)
    
    # Attack
    if damage_queue:
        attacker_slots = np.array([d[1] for d in damage_queue], dtype=np.int32)
        eff_attacks = np.array([d[2] for d in damage_queue], dtype=np.float32)
        apply_attack_cost(attacker_slots, eff_attacks)
    
    # Sound
    if sound_emit_queue:
        slots = np.array([s[0] for s in sound_emit_queue], dtype=np.int32)
        S_emit = np.array([s[2] for s in sound_emit_queue], dtype=np.float32)
        apply_sound_cost(slots, S_emit)
    
    # Active stealth
    if stealth_queue:
        slots = np.array([s[0] for s in stealth_queue], dtype=np.int32)
        boost = np.array([s[1] for s in stealth_queue], dtype=np.float32)
        apply_stealth_cost(slots, boost)
    
    # Energy transfer
    if transfer_queue:
        actor_slots = np.array([t[0] for t in transfer_queue], dtype=np.int32)
        transfer_amt = np.array([t[2] for t in transfer_queue], dtype=np.float32)
        apply_transfer_cost(actor_slots, transfer_amt)
    
    # Clamp energy
    np.clip(energy, 0.0, traits_shared[:, T_MAX_ENERGY], out=energy)

# ── Feeding ──────────────────────────────────────────────────────────────
def feed_from_herb(slot: int, herb_idx: int, herb_ep_val: float, is_herb: bool) -> tuple[float, float]:
    """
    Feed from herb. Returns (energy_gained, herb_ep_consumed).
    """
    rate = BASE_FEED_RATE * M_F[slot] * infant_factor[slot]
    consumed = min(herb_ep_val, rate)
    efficiency = 1.0 if is_herb else CARNIVORE_HERB_EFF
    gained = consumed * efficiency
    return gained, consumed

def feed_from_carcass(slot: int, carc_idx: int, carc_ep_val: float) -> tuple[float, float]:
    """
    Feed from carcass (carnivores only). Returns (energy_gained, carc_ep_consumed).
    """
    rate = BASE_FEED_RATE * M_F[slot] * infant_factor[slot]
    consumed = min(carc_ep_val, rate)
    bite_eff = traits_carn[slot, TC_BITE_EFF] * infant_factor[slot] / 100.0
    gained = consumed * bite_eff
    return gained, consumed

# ── Attack Energy Gain ───────────────────────────────────────────────────
def attack_energy_gain(slot: int, damage_dealt: float) -> float:
    """Energy gained from successful attack."""
    bite_eff = traits_carn[slot, TC_BITE_EFF] * infant_factor[slot] / 100.0
    return damage_dealt * bite_eff

# ── Energy Transfer Gain ─────────────────────────────────────────────────
def apply_transfer_gain(target_slots: np.ndarray, amounts: np.ndarray):
    """Apply energy gain to transfer targets."""
    energy[target_slots] += amounts
    np.clip(energy[target_slots], 0.0, traits_shared[target_slots, T_MAX_ENERGY], out=energy[target_slots])

# ── Reproduction Cost ────────────────────────────────────────────────────
def can_afford_reproduction(slot: int) -> bool:
    return energy[slot] >= REPRODUCE_COST

def deduct_reproduction_cost(slot_A: int, slot_B: int):
    energy[slot_A] -= REPRODUCE_COST
    energy[slot_B] -= REPRODUCE_COST