"""
FastAPI Server — WebSocket Dashboard, REST API
Shared snapshot (lock-free reads), Cloudflare tunnel compatible.
"""
import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import DRIVE_BASE, API_PORT, WS_PUSH_INTERVAL
from .tick_loop import (latest_snapshot, snapshot_version, snapshot_lock,
                       control_command, control_lock, sim_paused, sim_stop,
                       sim_speed, turn, _spawn_progenitors)
from .animal_model import (alive, species, x_pos, y_pos, energy, health, fatigue,
                          age, animal_ids, id_to_slot, generation,
                          traits_shared, count_herbivores, count_carnivores)

# ── FastAPI App ──────────────────────────────────────────────────────────
app = FastAPI(title="ENV Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (served from /static)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── WebSocket Manager ────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, data: dict):
        """Broadcast to all connected clients."""
        dead = []
        async with self._lock:
            for conn in self.active_connections:
                try:
                    await conn.send_json(data)
                except:
                    dead.append(conn)
            for d in dead:
                self.active_connections.remove(d)

manager = ConnectionManager()

# ── WebSocket Endpoint ───────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial snapshot
        with snapshot_lock:
            if latest_snapshot:
                await websocket.send_json(latest_snapshot)
        
        # Keep connection alive, listen for client messages
        while True:
            try:
                msg = await websocket.receive_text()
                # Handle client messages if needed
            except WebSocketDisconnect:
                break
            except:
                break
    finally:
        await manager.disconnect(websocket)

# ── Background Task: Push Snapshots ──────────────────────────────────────
async def snapshot_pusher():
    """Push latest snapshot to WebSocket clients periodically."""
    last_version = 0
    while True:
        await asyncio.sleep(1.0 / (WS_PUSH_INTERVAL / 10.0))  # Match sim speed roughly
        
        with snapshot_lock:
            snap = latest_snapshot
            ver = snapshot_version
        
        if snap and ver != last_version:
            await manager.broadcast(snap)
            last_version = ver

# ── REST API Endpoints ───────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    """Current simulation status."""
    with snapshot_lock:
        snap = latest_snapshot.copy() if latest_snapshot else {}
    snap["sim_speed"] = sim_speed
    snap["paused"] = sim_paused.is_set()
    snap["turn"] = turn
    return snap

@app.get("/api/world")
async def get_world():
    """World summary."""
    return {
        "herbivores": count_herbivores(),
        "carnivores": count_carnivores(),
        "total": count_herbivores() + count_carnivores(),
        "turn": turn,
    }

@app.get("/api/animals")
async def get_animals(
    species_filter: Optional[str] = Query(None, regex="^(herbivore|carnivore)$"),
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0)
):
    """List animals with pagination."""
    herb_idx = np.where(alive & species)[0] if species_filter != "carnivore" else np.array([], dtype=np.int32)
    carn_idx = np.where(alive & ~species)[0] if species_filter != "herbivore" else np.array([], dtype=np.int32)
    
    all_idx = np.concatenate([herb_idx, carn_idx])
    all_idx = all_idx[offset:offset+limit]
    
    animals = []
    for idx in all_idx:
        is_herb = bool(species[idx])
        animals.append({
            "id": animal_ids[idx],
            "species": "herbivore" if is_herb else "carnivore",
            "x": float(x_pos[idx]),
            "y": float(y_pos[idx]),
            "energy": float(energy[idx]),
            "health": float(health[idx]),
            "fatigue": float(fatigue[idx]),
            "age": int(age[idx]),
            "generation": int(generation[idx]),
        })
    
    return {"animals": animals, "total": int(np.sum(alive))}

@app.get("/api/animal/{animal_id}")
async def get_animal(animal_id: str):
    """Get detailed animal state."""
    slot = id_to_slot.get(animal_id)
    if slot is None or not alive[slot]:
        return {"error": "Animal not found"}, 404
    
    is_herb = bool(species[slot])
    return {
        "id": animal_id,
        "species": "herbivore" if is_herb else "carnivore",
        "slot": slot,
        "position": {"x": float(x_pos[slot]), "y": float(y_pos[slot])},
        "vitals": {
            "energy": float(energy[slot]),
            "max_energy": float(traits_shared[slot, 1]),
            "health": float(health[slot]),
            "max_health": float(traits_shared[slot, 0]),
            "fatigue": float(fatigue[slot]),
            "max_fatigue": float(traits_shared[slot, 4]),
        },
        "age": int(age[slot]),
        "lifespan": float(traits_shared[slot, 2]),
        "generation": int(generation[slot]),
        "traits": {
            "speed": float(traits_shared[slot, 3]),
            "metabolism": float(traits_shared[slot, 5]),
            "detection_efficiency": float(traits_shared[slot, 6]),
            "sound_sensitivity": float(traits_shared[slot, 7]),
            "sound_power": float(traits_shared[slot, 8]),
        } | ({
            "attack_power": float(traits_carn[slot, 0]),
            "stealth": float(traits_carn[slot, 1]),
            "bite_efficiency": float(traits_carn[slot, 2]),
        } if not is_herb else {
            "camouflage_efficiency": float(traits_herb[slot, 0]),
            "defence": float(traits_herb[slot, 1]),
            "active_stealth_power": float(traits_herb[slot, 2]),
        }),
    }

@app.post("/api/control/pause")
async def pause_sim():
    sim_paused.set()
    return {"status": "paused"}

@app.post("/api/control/resume")
async def resume_sim():
    sim_paused.clear()
    return {"status": "running"}

@app.post("/api/control/speed")
async def set_speed(value: int = Body(..., embed=True)):
    global sim_speed
    sim_speed = max(1, min(1000, value))
    return {"sim_speed": sim_speed}

@app.post("/api/control/stop")
async def stop_sim():
    sim_stop.set()
    return {"status": "stopped"}

@app.post("/api/control/seed")
async def set_seed(value: int = Body(..., embed=True)):
    from .rng import reseed
    reseed(value)
    return {"seed": value}

@app.post("/api/control/restart")
async def restart_sim():
    """Restart simulation with new progenitors."""
    sim_stop.set()
    await asyncio.sleep(0.5)
    sim_stop.clear()
    sim_paused.clear()
    
    # Reset will be handled by tick_loop restarting
    # For now just reset state
    from .animal_model import reset_arrays
    from .world import init_world
    reset_arrays()
    init_world(seed=42)
    _spawn_progenitors(20, 5)
    
    return {"status": "restarted"}

# ── Log Query Endpoints ──────────────────────────────────────────────────
@app.get("/api/logs/runs")
async def list_runs():
    """List available run directories."""
    runs = []
    if os.path.exists(DRIVE_BASE):
        for name in sorted(os.listdir(DRIVE_BASE), reverse=True):
            path = os.path.join(DRIVE_BASE, name)
            if os.path.isdir(path):
                runs.append({"run_id": name, "path": path})
    return {"runs": runs}

@app.get("/api/logs/{run_id}/ids")
async def get_ids_log(run_id: str, limit: int = Query(100, ge=1, le=10000)):
    """Read ids.jsonl for a run."""
    path = os.path.join(DRIVE_BASE, run_id, "ids.jsonl")
    if not os.path.exists(path):
        return {"error": "Not found"}, 404
    
    records = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                records.append(json.loads(line))
            except:
                pass
    return {"records": records}

@app.get("/api/logs/{run_id}/dynamic")
async def get_dynamic_log(
    run_id: str,
    turn: Optional[int] = Query(None),
    entity_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=10000)
):
    """Read dynamic.jsonl with optional filters."""
    path = os.path.join(DRIVE_BASE, run_id, "dynamic.jsonl")
    if not os.path.exists(path):
        return {"error": "Not found"}, 404
    
    records = []
    with open(path, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)
                if turn is not None and rec.get("turn") != turn:
                    continue
                if entity_id is not None and rec.get("id") != entity_id and rec.get("child_id") != entity_id:
                    continue
                records.append(rec)
                if len(records) >= limit:
                    break
            except:
                pass
    return {"records": records}

# ── Frontend ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard HTML."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("""
    <html><body>
    <h1>ENV Simulation Dashboard</h1>
    <p>Static files not found. Place index.html in code/systems/static/</p>
    </body></html>
    """)

# ── Startup ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(snapshot_pusher())

# ── Main ─────────────────────────────────────────────────────────────────
def run_server():
    """Run uvicorn server."""
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")

if __name__ == "__main__":
    run_server()