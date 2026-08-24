# ENV Simulation — Evolutionary Ecosystem in Colab

A 2D evolutionary ecosystem simulation with neural-network-driven animals, genetic inheritance, and real-time web dashboard.

## Quick Start (Google Colab)

1. Open `code/ENV_Simulation_Colab.ipynb` in Google Colab
2. Run all cells sequentially
3. Copy the Cloudflare Tunnel URL (e.g., `https://abc123.trycloudflare.com`)
4. Open in browser → live dashboard

## Architecture

```
code/
├── ENV_Simulation_Colab.ipynb    # Colab runner notebook
├── systems/                      # Core simulation modules
│   ├── config.py                 # All constants (single source of truth)
│   ├── rng.py                    # Deterministic RNG
│   ├── animal_model.py           # NumPy SoA animal state
│   ├── genome_traits.py          # Flat binary genome, vectorized ops
│   ├── mutation_repro.py         # Inheritance + persistent mate-seeking
│   ├── brain_nn.py               # Batched einsum NN (32→16)
│   ├── detection.py              # Numba JIT cone detection
│   ├── sound.py                  # Fixed-array inbox (no heap)
│   ├── energy.py                 # Unified Phase 5 costs
│   ├── fatigue.py                # Normalized to endurance
│   ├── memory.py                 # Integer blend
│   ├── world.py                  # Cell-based obstacles, herbs, carcasses
│   ├── tick_loop.py              # Unified 9-phase tick loop
│   ├── logging_module.py         # Buffered JSONL + binary checkpoints
│   ├── server.py                 # FastAPI + WebSocket dashboard
│   └── static/index.html         # Real-time dashboard UI
└── py-opt docs/                  # Design documentation
```

## Key Features

- **Pure NumPy hot path** — zero Python objects in tick loop
- **Numba JIT detection** — 10-50× faster than Python
- **Deterministic RNG** — reproducible runs with same seed
- **Compressed checkpoints** — zlib + pickle, resume from Drive
- **Unified energy accounting** — all costs in single Phase 5 pass
- **Reduced NN** — 32 inputs / 16 outputs / ~800 params (2× faster)
- **Persistent mate-seeking** — biologically plausible reproduction
- **Cell-based obstacles** — local density self-regulation
- **Carcass scent** — long-range detection via sound system

## Configuration

All tunable parameters in `systems/config.py`:
- Population caps, world size, grid resolution
- NN architecture, mutation rates
- Energy costs, fatigue curves
- Reproduction, infant debuff
- Logging modes, checkpoint intervals

## Output Structure (Google Drive)

```
/content/drive/MyDrive/envsim/
├── 2026-01-15_143022/          # Run directory
│   ├── ids.jsonl               # Entity registry (schema v2)
│   ├── dynamic.jsonl           # Per-turn events
│   └── checkpoints/
│       ├── turn_00000500.pkl.zst
│       └── turn_00001000.pkl.zst
└── .numba_cache/               # Numba JIT cache
```

## Dashboard Features

- Live world map (Canvas)
- Population charts
- Animal inspector (click any entity)
- Pause/resume/speed controls
- REST API for log queries
- WebSocket real-time updates

## Performance Targets (Colab Free Tier)

| Population | Target Turns/sec |
|------------|------------------|
| 200        | 100+             |
| 500        | 60+              |
| 1000       | 30+              |
| 2000       | 15+              |

## Validation

Run tests in Colab:
```python
!pip install pytest
%run -m pytest test_validation.py -v
```

## License

MIT