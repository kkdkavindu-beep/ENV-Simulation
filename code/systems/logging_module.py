"""
Logging System — Buffered JSONL to Google Drive
Schema versioning, event-only mode, binary genome checkpoints.
"""
import json
import orjson
import os
import threading
from .config import DRIVE_BASE, RUN_ID, LOG_MODE, LOG_SAMPLE_RATE
from .animal_model import (alive, species, x_pos, y_pos, energy, health, fatigue, age,
                          animal_ids, generation, traits_shared, traits_carn, traits_herb)

# ── Log Writers ──────────────────────────────────────────────────────────
class LogWriter:
    def __init__(self, path: str, buffer_lines: int = 2000):
        self._f = open(path, 'a', encoding='utf-8', buffering=8 * 1024 * 1024)
        self._buf = []
        self._cap = buffer_lines
        self._lock = threading.Lock()
        self._lines_written = 0
    
    def write(self, record: dict):
        with self._lock:
            self._buf.append(orjson.dumps(record).decode())
            if len(self._buf) >= self._cap:
                self._flush()
    
    def _flush(self):
        if self._buf:
            self._f.write('\n'.join(self._buf) + '\n')
            self._lines_written += len(self._buf)
            self._buf.clear()
    
    def flush(self):
        with self._lock:
            self._flush()
            self._f.flush()
    
    def close(self):
        self.flush()
        self._f.close()

# Global writers (initialized in Colab notebook)
ids_writer = None
dynamic_writer = None

# ── Schema Version ───────────────────────────────────────────────────────
SCHEMA_VERSION = 2

def write_schema_headers():
    """Write schema version as first line of each log file."""
    if ids_writer:
        ids_writer.write({"schema_version": SCHEMA_VERSION, "type": "ids", "run_id": RUN_ID})
        ids_writer._flush()
    if dynamic_writer:
        dynamic_writer.write({"schema_version": SCHEMA_VERSION, "type": "dynamic", "run_id": RUN_ID})
        dynamic_writer._flush()

# ── IDS.jsonl (Entity Registry) ─────────────────────────────────────────
def log_entity_creation(entity_id: str, entity_type: str, x: float, y: float,
                        genome_hash: str = None, generation: int = 0,
                        traits: dict = None):
    """Log new entity to ids.jsonl (write-once)."""
    record = {
        "turn": turn,
        "id": entity_id,
        "type": entity_type,  # "herbivore", "carnivore", "obstacle", "herb", "carcass"
        "x": x, "y": y,
        "generation": generation,
    }
    if genome_hash:
        record["genome_hash"] = genome_hash
    if traits:
        record["traits"] = traits
    
    if ids_writer:
        ids_writer.write(record)
        ids_writer._flush()  # Immediate flush for registry

# ── Dynamic.jsonl (Per-Turn Events) ─────────────────────────────────────
# turn is imported from tick_loop at runtime; do not redeclare here.

def should_log_entity(slot: int) -> bool:
    """Determine if entity should be logged this turn based on LOG_MODE."""
    if LOG_MODE == "full":
        return True
    elif LOG_MODE == "sampled":
        return turn % LOG_SAMPLE_RATE == 0
    elif LOG_MODE == "event_only":
        return False  # Only log on events
    return False

def log_tick(current_turn: int):
    """Log per-turn state for sampled entities."""
    from .tick_loop import turn as _tick_turn
    global turn
    turn = _tick_turn
    
    if LOG_MODE == "event_only":
        return  # No periodic logging
    
    alive_idx = np.where(alive)[0]
    for slot in alive_idx:
        if not should_log_entity(slot):
            continue
        
        is_herb = bool(species[slot])
        record = {
            "turn": turn,
            "id": animal_ids[slot],
            "species": "herbivore" if is_herb else "carnivore",
            "x": float(x_pos[slot]),
            "y": float(y_pos[slot]),
            "energy": float(energy[slot]),
            "health": float(health[slot]),
            "fatigue": float(fatigue[slot]),
            "age": int(age[slot]),
            "generation": int(generation[slot]),
        }
        
        if dynamic_writer:
            dynamic_writer.write(record)

def log_birth(child_id: str, parent_a: str, parent_b: str, x: float, y: float):
    """Log birth event."""
    record = {
        "turn": turn,
        "event": "birth",
        "child_id": child_id,
        "parent_a": parent_a,
        "parent_b": parent_b,
        "x": x, "y": y,
    }
    if dynamic_writer:
        dynamic_writer.write(record)

def log_death(entity_id: str, current_turn: int, cause: str):
    """Log death event."""
    record = {
        "turn": current_turn,
        "event": "death",
        "id": entity_id,
        "cause": cause,  # "starvation", "injury", "old_age"
    }
    if dynamic_writer:
        dynamic_writer.write(record)

def log_event(event_type: str, **kwargs):
    """Log generic event."""
    record = {"turn": turn, "event": event_type}
    record.update(kwargs)
    if dynamic_writer:
        dynamic_writer.write(record)

def log_combat(attacker_id: str, target_id: str, damage: float, energy_gained: float):
    log_event("combat", attacker=attacker_id, target=target_id,
              damage=damage, energy_gained=energy_gained)

def log_feeding(entity_id: str, source_type: str, source_id: str, energy_gained: float):
    log_event("feed", entity=entity_id, source_type=source_type,
              source_id=source_id, energy_gained=energy_gained)

def log_reproduction(parent_a: str, parent_b: str, child_id: str):
    log_event("reproduction", parent_a=parent_a, parent_b=parent_b, child=child_id)

def log_sound_emission(emitter_id: str, signal_type: int, strength: float):
    if LOG_MODE != "event_only":  # Too verbose for event-only
        return
    log_event("sound_emit", emitter=emitter_id, type=signal_type, strength=strength)

# Import needed
import numpy as np