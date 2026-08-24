"""
Simulation Tick Loop — Unified Phases, Numba Detection
"""
import time
import numpy as np
import threading
from .config import *
from .rng import get_rng, reseed
from .animal_model import (alive, species, x_pos, y_pos, energy, health, fatigue, age,
                          memory, traits_shared, traits_carn, traits_herb,
                          M_F, infant_factor, mate_seek_timer, generation,
                          genomes, genome_valid, animal_ids, id_to_slot,
                          animal_grid, herb_grid, obstacle_grid, carcass_grid,
                          herb_positions, obstacle_positions,
                          allocate_slot, release_slot, register_animal,
                          get_dead_mask, count_herbivores, count_carnivores,
                          last_scan, last_decoded, active_stealth_boost,
                          W1_h, b1_h, W2_h, b2_h, W3_h, b3_h,
                          W1_c, b1_c, W2_c, b2_c, W3_c, b3_c,
                          sound_inbox)
from .world import (update_obstacles_vectorized, update_herbs_vectorized,
                   update_carcasses_vectorized, spawn_carcass, rebuild_grids)
from .genome_traits import (compute_infant_factor_vectorized, populate_trait_cache,
                           load_weights_from_genome)
from .mutation_repro import (create_offspring, find_reproduction_pairs, update_mate_seek)
from .brain_nn import (run_nn_tick, decode_outputs, set_weight_tensors,
                      get_memory_read_idx, update_memory_read_idx)
from .detection import (run_detection_batch, resolve_target_lock)
from .sound import (broadcast_sounds_vectorized, add_carcass_scent_to_inbox,
                   build_sound_inputs_batch)
from .energy import (apply_all_costs, feed_from_herb, feed_from_carcass,
                    attack_energy_gain, apply_transfer_gain)
from .fatigue import (recover_fatigue, compute_M_F, accumulate_fatigue)
from .memory import blend_memory
from .logging_module import (log_tick, log_birth, log_death, log_event,
                            ids_writer, dynamic_writer)

# ── Inter-thread Communication ───────────────────────────────────────────
latest_snapshot = {}
snapshot_version = 0
snapshot_lock = threading.Lock()

control_command = None
control_lock = threading.Lock()

sim_paused = threading.Event()
sim_stop = threading.Event()
sim_speed = TARGET_SIM_SPEED

turn = 0
active_detect_slots = []

# ── Initialize Weight Tensors ────────────────────────────────────────────
def init_weight_tensors():
    set_weight_tensors(W1_h, b1_h, W2_h, b2_h, W3_h, b3_h,
                       W1_c, b1_c, W2_c, b2_c, W3_c, b3_c)

# ── Control Handler ──────────────────────────────────────────────────────
def _drain_control():
    global sim_speed, control_command
    with control_lock:
        cmd = control_command
        control_command = None
    
    if cmd is None:
        return
    
    action = cmd.get("action")
    if action == "pause":
        sim_paused.set()
    elif action == "resume":
        sim_paused.clear()
    elif action == "speed":
        sim_speed = int(cmd.get("value", TARGET_SIM_SPEED))
    elif action == "stop":
        sim_stop.set()
    elif action == "seed":
        reseed(cmd.get("value", DEFAULT_SEED))

# ── State Snapshot ───────────────────────────────────────────────────────
def _update_latest_snapshot():
    global latest_snapshot, snapshot_version
    
    alive_mask = alive
    herb_idx = np.where(alive_mask & species)[0]
    carn_idx = np.where(alive_mask & ~species)[0]
    
    snap = {
        "turn": turn,
        "version": snapshot_version + 1,
        "world": {
            "herbivore_count": len(herb_idx),
            "carnivore_count": len(carn_idx),
            "herb_count": int(np.sum(herb_alive)),
            "carcass_count": int(np.sum(carc_alive)),
            "obstacle_count": int(np.sum(obs_alive)),
        },
        "animals": _build_animal_snapshots(herb_idx, carn_idx),
        "herbs": _build_herb_snapshots(),
        "carcasses": _build_carcass_snapshots(),
        "events": _get_recent_events(),
    }
    
    with snapshot_lock:
        latest_snapshot = snap
        snapshot_version += 1

def _build_animal_snapshots(herb_idx, carn_idx):
    """Build minimal animal data for UI."""
    snaps = []
    for idx in herb_idx:
        snaps.append({
            "id": animal_ids[idx],
            "species": "herbivore",
            "x": float(x_pos[idx]),
            "y": float(y_pos[idx]),
            "energy": float(energy[idx]),
            "health": float(health[idx]),
            "fatigue": float(fatigue[idx]),
            "age": int(age[idx]),
            "generation": int(generation[idx]),
        })
    for idx in carn_idx:
        snaps.append({
            "id": animal_ids[idx],
            "species": "carnivore",
            "x": float(x_pos[idx]),
            "y": float(y_pos[idx]),
            "energy": float(energy[idx]),
            "health": float(health[idx]),
            "fatigue": float(fatigue[idx]),
            "age": int(age[idx]),
            "generation": int(generation[idx]),
        })
    return snaps

def _build_herb_snapshots():
    snaps = []
    for idx in np.where(herb_alive)[0]:
        snaps.append({
            "id": f"PL{idx:06d}",
            "x": float(herb_x[idx]),
            "y": float(herb_y[idx]),
            "ep": float(herb_ep[idx]),
        })
    return snaps

def _build_carcass_snapshots():
    snaps = []
    for idx in np.where(carc_alive)[0]:
        snaps.append({
            "id": f"CA{idx:06d}",
            "x": float(carc_x[idx]),
            "y": float(carc_y[idx]),
            "ep": float(carc_ep[idx]),
        })
    return snaps

def _get_recent_events():
    # Return recent logged events (placeholder)
    return []

# ── Checkpointing ────────────────────────────────────────────────────────
def checkpoint_to_drive(turn_num: int):
    """Save checkpoint to Google Drive."""
    import pickle, zlib, os
    from .config import DRIVE_BASE, RUN_ID
    
    chk_dir = os.path.join(DRIVE_BASE, RUN_ID, "checkpoints")
    os.makedirs(chk_dir, exist_ok=True)
    chk_path = os.path.join(chk_dir, f"turn_{turn_num:08d}.pkl.zst")
    
    alive_idx = np.where(alive)[0]
    
    state = {
        "meta": {"turn": turn_num, "run_id": RUN_ID, "seed": DEFAULT_SEED, "version": 2},
        "sim": {
            "alive": alive[alive_idx], "species": species[alive_idx],
            "x_pos": x_pos[alive_idx], "y_pos": y_pos[alive_idx],
            "energy": energy[alive_idx], "health": health[alive_idx],
            "fatigue": fatigue[alive_idx], "age": age[alive_idx],
            "memory": memory[alive_idx], "trait_cache": traits_shared[alive_idx],
            "M_F": M_F[alive_idx], "infant_factor": infant_factor[alive_idx],
            "mate_seek_timer": mate_seek_timer[alive_idx],
            "generation": generation[alive_idx],
            "alive_indices": alive_idx,
        },
        "nn": {
            "W1_h": W1_h[alive_idx & species], "b1_h": b1_h[alive_idx & species],
            "W2_h": W2_h[alive_idx & species], "b2_h": b2_h[alive_idx & species],
            "W3_h": W3_h[alive_idx & species], "b3_h": b3_h[alive_idx & species],
            "W1_c": W1_c[alive_idx & ~species], "b1_c": b1_c[alive_idx & ~species],
            "W2_c": W2_c[alive_idx & ~species], "b2_c": b2_c[alive_idx & ~species],
            "W3_c": W3_c[alive_idx & ~species], "b3_c": b3_c[alive_idx & ~species],
        },
        "genomes": {"data": genomes[alive_idx], "valid": genome_valid[alive_idx], "indices": alive_idx},
        "ids": {"animal_ids": [animal_ids[i] for i in alive_idx],
                "next_herb_id": next_herb_id, "next_carn_id": next_carn_id},
        "world": {
            "herb_x": herb_x[herb_alive], "herb_y": herb_y[herb_alive], "herb_ep": herb_ep[herb_alive],
            "carc_x": carc_x[carc_alive], "carc_y": carc_y[carc_alive], "carc_ep": carc_ep[carc_alive],
            "carc_age": carc_age[carc_alive],
        },
    }
    
    data = pickle.dumps(state, protocol=4)
    compressed = zlib.compress(data, level=3)
    
    with open(chk_path, 'wb') as f:
        f.write(compressed)
    
    print(f"💾 Checkpoint: turn {turn_num} ({len(compressed)/1e6:.1f} MB)")

# ── Progenitor Spawning ──────────────────────────────────────────────────
def _spawn_progenitors(n_herb: int, n_carn: int):
    from .genome_traits import create_progenitor_genome
    
    for _ in range(n_herb):
        slot = allocate_slot()
        if slot is None: break
        genome = create_progenitor_genome(True)
        populate_trait_cache(slot, genome, True)
        load_weights_from_genome(slot, genome, True)
        register_animal(slot, True, genome,
                       get_rng().uniform(0, WORLD_SIZE), get_rng().uniform(0, WORLD_SIZE),
                       0.5 * traits_shared[slot, T_MAX_ENERGY],
                       int(0.10 * traits_shared[slot, T_LIFESPAN]),
                       None, None, None, genome, 0)
    
    for _ in range(n_carn):
        slot = allocate_slot()
        if slot is None: break
        genome = create_progenitor_genome(False)
        populate_trait_cache(slot, genome, False)
        load_weights_from_genome(slot, genome, False)
        register_animal(slot, False, genome,
                       get_rng().uniform(0, WORLD_SIZE), get_rng().uniform(0, WORLD_SIZE),
                       0.5 * traits_shared[slot, T_MAX_ENERGY],
                       int(0.10 * traits_shared[slot, T_LIFESPAN]),
                       None, None, None, genome, 0)

# ── Main Tick Loop ───────────────────────────────────────────────────────
def tick_loop():
    """Main simulation loop — runs in background thread."""
    global turn, active_detect_slots
    
    init_weight_tensors()
    
    while not sim_stop.is_set():
        if sim_paused.is_set():
            time.sleep(0.05)
            continue
        
        _drain_control()
        
        t0 = time.perf_counter()
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 0: World Update
        # ═══════════════════════════════════════════════════════════════
        update_obstacles_vectorized(turn)
        update_herbs_vectorized()
        update_carcasses_vectorized()
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 1: Pre-Processing
        # ═══════════════════════════════════════════════════════════════
        alive_idx = np.where(alive)[0]
        
        recover_fatigue(alive_idx)
        
        # Living cost
        energy[alive_idx] -= LIVING_COST_FACTOR * traits_shared[alive_idx, T_METABOLISM]
        
        # Fatigue multiplier
        compute_M_F(alive_idx)
        
        # Infant factor
        compute_infant_factor_vectorized(alive_idx)
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Sensing
        # ═══════════════════════════════════════════════════════════════
        # 2a. Active Detection
        scan_results, detect_fired, active_next = run_detection_batch(
            active_detect_slots, x_pos, y_pos,
            animal_grid, herb_grid, obstacle_grid,
            traits_shared, traits_carn, traits_herb,
            M_F, infant_factor, species, alive
        )
        active_detect_slots = active_next
        
        # 2b. Sound Broadcast
        sound_inbox.fill(0.0)
        broadcast_sounds_vectorized(
            np.array([], dtype=np.int32),  # emitters from previous tick
            np.array([]), np.array([]),
            x_pos, y_pos, animal_grid,
            traits_shared, M_F, infant_factor, species, alive
        )
        
        # 2c. Carcass Scent
        add_carcass_scent_to_inbox(carcass_grid, x_pos, y_pos,
                                   sound_inbox, alive, species, traits_shared)
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 3: Neural Network
        # ═══════════════════════════════════════════════════════════════
        herb_idx = np.where(alive & species)[0]
        carn_idx = np.where(alive & ~species)[0]
        
        # Sound inputs from inbox
        sound_h = build_sound_inputs_batch(herb_idx) if len(herb_idx) > 0 else np.zeros((0, 8), dtype=np.float32)
        sound_c = build_sound_inputs_batch(carn_idx) if len(carn_idx) > 0 else np.zeros((0, 8), dtype=np.float32)
        
        Z3_h, Z3_c = run_nn_tick(herb_idx, carn_idx, scan_results, sound_h, sound_c)
        
        decoded_h = decode_outputs(Z3_h, is_herb=True) if len(herb_idx) > 0 else {}
        decoded_c = decode_outputs(Z3_c, is_herb=False) if len(carn_idx) > 0 else {}
        
        # Store decoded for next tick's detection
        if len(herb_idx) > 0:
            for i, slot in enumerate(herb_idx):
                last_decoded[slot] = {k: v[i] for k, v in decoded_h.items()}
                # Pre-decode detection params
                if last_decoded[slot].get("detect_active", False):
                    theta, R, alpha = decode_outputs(np.array([[0]*16]), True)  # placeholder
                    # Actually decode from decoded_h
                    dir_n = decoded_h["detect_dir"][i]
                    range_n = decoded_h["detect_range"][i]
                    angle_n = decoded_h["detect_angle"][i]
                    from .brain_nn import decode_detection_params
                    theta, R, alpha = decode_detection_params(
                        np.array([dir_n]), np.array([range_n]), np.array([angle_n])
                    )
                    last_decoded[slot]["detect_theta"] = float(theta[0])
                    last_decoded[slot]["detect_R"] = float(R[0])
                    last_decoded[slot]["detect_alpha"] = float(alpha[0])
        
        if len(carn_idx) > 0:
            for i, slot in enumerate(carn_idx):
                last_decoded[slot] = {k: v[i] for k, v in decoded_c.items()}
                if last_decoded[slot].get("detect_active", False):
                    dir_n = decoded_c["detect_dir"][i]
                    range_n = decoded_c["detect_range"][i]
                    angle_n = decoded_c["detect_angle"][i]
                    from .brain_nn import decode_detection_params
                    theta, R, alpha = decode_detection_params(
                        np.array([dir_n]), np.array([range_n]), np.array([angle_n])
                    )
                    last_decoded[slot]["detect_theta"] = float(theta[0])
                    last_decoded[slot]["detect_R"] = float(R[0])
                    last_decoded[slot]["detect_alpha"] = float(alpha[0])
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 4: Action Collection
        # ═══════════════════════════════════════════════════════════════
        feed_queue = []
        reproduce_queue = []
        transfer_queue = []
        damage_queue = []
        stealth_queue = []
        sound_emit_queue = []
        move_queue = []
        memory_write_queue = []
        fatigue_delta = np.zeros(MAX_ANIMALS, dtype=np.float32)
        
        # Herbivore actions
        if len(herb_idx) > 0:
            for i, slot in enumerate(herb_idx):
                _collect_actions_herb(slot, i, decoded_h,
                                      feed_queue, reproduce_queue,
                                      transfer_queue, stealth_queue,
                                      sound_emit_queue, move_queue,
                                      memory_write_queue, fatigue_delta)
        
        # Carnivore actions
        if len(carn_idx) > 0:
            for i, slot in enumerate(carn_idx):
                _collect_actions_carn(slot, i, decoded_c,
                                      feed_queue, reproduce_queue,
                                      transfer_queue, damage_queue,
                                      sound_emit_queue, move_queue,
                                      memory_write_queue, fatigue_delta)
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 5: APPLY ALL COSTS
        # ═══════════════════════════════════════════════════════════════
        apply_all_costs(move_queue, detect_fired, damage_queue,
                        sound_emit_queue, stealth_queue, transfer_queue,
                        fatigue_delta)
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 6: Resolve Interactions
        # ═══════════════════════════════════════════════════════════════
        # Active stealth boost
        for slot, boost in stealth_queue:
            active_stealth_boost[slot] = boost
        
        # Combat
        for target_slot, attacker_slot, eff_attack in damage_queue:
            if not alive[target_slot] or not alive[attacker_slot]:
                continue
            # Defense
            if species[target_slot]:  # herbivore target
                defence = traits_herb[target_slot, TH_DEFENCE] * infant_factor[target_slot]
                defence = min(defence, 99.0)
                dmg = eff_attack * (1.0 - defence / 100.0)
            else:  # carnivore target
                defence = 0.0  # carnivores have no defence trait
                dmg = eff_attack
            
            health[target_slot] -= dmg
            
            # Attacker gains energy
            gain = attack_energy_gain(attacker_slot, dmg)
            energy[attacker_slot] = min(energy[attacker_slot] + gain,
                                        traits_shared[attacker_slot, T_MAX_ENERGY])
        
        # Feeding
        for slot in feed_queue:
            if not alive[slot]:
                continue
            is_herb = species[slot]
            row = int(y_pos[slot] // CELL_SIZE)
            col = int(x_pos[slot] // CELL_SIZE)
            
            # Try herb first
            fed = False
            for hid in list(herb_grid[row][col]):
                h_idx = int(hid[2:])
                if herb_ep[h_idx] > 0:
                    gained, consumed = feed_from_herb(slot, h_idx, herb_ep[h_idx], is_herb)
                    energy[slot] = min(energy[slot] + gained, traits_shared[slot, T_MAX_ENERGY])
                    herb_ep[h_idx] -= consumed
                    if herb_ep[h_idx] <= 0:
                        herb_alive[h_idx] = False
                        herb_grid[row][col].discard(hid)
                        herb_positions.pop(hid, None)
                    fed = True
                    break
            
            # Carnivore: try carcass
            if not fed and not is_herb:
                for cid in list(carcass_grid[row][col]):
                    c_idx = int(cid[2:])
                    if carc_ep[c_idx] > 0:
                        gained, consumed = feed_from_carcass(slot, c_idx, carc_ep[c_idx])
                        energy[slot] = min(energy[slot] + gained, traits_shared[slot, T_MAX_ENERGY])
                        carc_ep[c_idx] -= consumed
                        if carc_ep[c_idx] <= 0:
                            carc_alive[c_idx] = False
                            carcass_grid[row][col].discard(cid)
                        fed = True
                        break
        
        # Energy transfer
        for actor_slot, target_slot, amount in transfer_queue:
            if alive[actor_slot] and alive[target_slot] and species[actor_slot] == species[target_slot]:
                apply_transfer_gain(np.array([target_slot]), np.array([amount]))
        
        # Reproduction
        update_mate_seek(herb_idx, carn_idx, decoded_h, decoded_c)
        pairs = find_reproduction_pairs(np.zeros(MAX_ANIMALS, dtype=np.bool_))  # uses mate_seek_timer
        for slot_A, slot_B in pairs:
            create_offspring(slot_A, slot_B)
        
        # Memory writes
        for slot, mem_idx, mem_val in memory_write_queue:
            if alive[slot]:
                blend_memory(np.array([slot]), np.array([mem_idx]), np.array([mem_val]))
        
        # ════════════════════════════════════════════════════════════════
        # Phase 7: State Update
        # ═══════════════════════════════════════════════════════════════
        np.clip(energy, 0.0, traits_shared[:, T_MAX_ENERGY], out=energy)
        np.clip(health, 0.0, traits_shared[:, T_HEALTH], out=health)
        
        accumulate_fatigue(fatigue_delta, alive_idx)
        
        age[alive_idx] += 1
        
        mate_seek_timer[alive_idx] = np.maximum(0, mate_seek_timer[alive_idx] - 1)
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 8: Death and Cleanup
        # ═══════════════════════════════════════════════════════════════
        dead_mask = get_dead_mask()
        dead_idx = np.where(dead_mask)[0]
        
        for slot in dead_idx:
            # Spawn carcass
            max_e = traits_shared[slot, T_MAX_ENERGY]
            spawn_carcass(slot, max_e)
            log_death(animal_ids[slot], turn, "starvation" if energy[slot] <= 0 else "injury" if health[slot] <= 0 else "old_age")
            release_slot(slot)
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 9: Logging + Snapshot + Checkpoint
        # ═══════════════════════════════════════════════════════════════
        log_tick(turn)
        
        if turn % PUSH_INTERVAL == 0:
            _update_latest_snapshot()
        
        if turn % CHECKPOINT_INTERVAL == 0:
            checkpoint_to_drive(turn)
        
        turn += 1
        
        # Speed control
        elapsed = time.perf_counter() - t0
        target = 1.0 / max(1, sim_speed)
        if elapsed < target:
            time.sleep(target - elapsed)

# ── Action Collection Helpers ────────────────────────────────────────────
def _collect_actions_herb(slot, i, decoded, feed_queue, reproduce_queue,
                          transfer_queue, stealth_queue, sound_emit_queue,
                          move_queue, memory_write_queue, fatigue_delta):
    # Movement
    speed = get_effective_speed(np.array([slot]))[0]
    dx = decoded["move_x"][i] * speed
    dy = decoded["move_y"][i] * speed
    new_x = max(0, min(WORLD_SIZE, x_pos[slot] + dx))
    new_y = max(0, min(WORLD_SIZE, y_pos[slot] + dy))
    move_queue.append((slot, new_x, new_y, dx, dy))
    
    # Sound emission
    if decoded["signal_active"][i]:
        sound_emit_queue.append((slot, decoded["signal_type"][i], decoded["signal_str"][i]))
    
    # Active stealth (action7)
    if decoded["action7"][i] > 0:
        boost = (decoded["action7"][i] / 255.0) * get_effective_stealth_power(np.array([slot]))[0]
        stealth_queue.append((slot, boost))
    
    # Feed
    if decoded["feed"][i]:
        feed_queue.append(slot)
    
    # Energy transfer
    if decoded["energy_transfer"][i]:
        target_slot = resolve_target_lock(decoded["target_lock"][i], last_scan[slot])
        if target_slot is not None and alive[target_slot] and species[target_slot]:
            transfer_queue.append((slot, target_slot, 
                                  min(energy[slot] * TRANSFER_FRACTION, TRANSFER_CAP)))
    
    # Mate seek handled in Phase 6
    
    # Memory write
    memory_write_queue.append((slot, decoded["mem_write_idx"][i], decoded["mem_write_val"][i]))

def _collect_actions_carn(slot, i, decoded, feed_queue, reproduce_queue,
                          transfer_queue, damage_queue, sound_emit_queue,
                          move_queue, memory_write_queue, fatigue_delta):
    # Movement
    speed = get_effective_speed(np.array([slot]))[0]
    dx = decoded["move_x"][i] * speed
    dy = decoded["move_y"][i] * speed
    new_x = max(0, min(WORLD_SIZE, x_pos[slot] + dx))
    new_y = max(0, min(WORLD_SIZE, y_pos[slot] + dy))
    move_queue.append((slot, new_x, new_y, dx, dy))
    
    # Sound emission
    if decoded["signal_active"][i]:
        sound_emit_queue.append((slot, decoded["signal_type"][i], decoded["signal_str"][i]))
    
    # Attack (action7)
    if decoded["action7"][i] > 0:
        target_slot = resolve_target_lock(decoded["target_lock"][i], last_scan[slot])
        if target_slot is not None and alive[target_slot] and not species[target_slot]:
            eff_attack = (decoded["action7"][i] / 255.0) * get_effective_attack(np.array([slot]))[0]
            if eff_attack > 0:
                damage_queue.append((target_slot, slot, eff_attack))
    
    # Feed
    if decoded["feed"][i]:
        feed_queue.append(slot)
    
    # Energy transfer
    if decoded["energy_transfer"][i]:
        target_slot = resolve_target_lock(decoded["target_lock"][i], last_scan[slot])
        if target_slot is not None and alive[target_slot] and not species[target_slot]:
            transfer_queue.append((slot, target_slot,
                                  min(energy[slot] * TRANSFER_FRACTION, TRANSFER_CAP)))
    
    # Mate seek handled in Phase 6
    
    # Memory write
    memory_write_queue.append((slot, decoded["mem_write_idx"][i], decoded["mem_write_val"][i]))

# Import needed for action collection
from .animal_model import get_effective_speed, get_effective_stealth_power, get_effective_attack