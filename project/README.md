# QuantumITS — Hybrid Quantum–Classical Intelligent Transportation

Course project for **Case Study 5: National Intelligent Transportation System**. The platform has to predict
congestion, recommend alternate routes and do both within **one second**. This project tests where small
quantum circuits help, and keeps every other stage classical.

| Module | Quantum method | File |
|---|---|---|
| Congestion prediction | Data re-uploading variational quantum classifier (4 qubits, 50 parameters), quantum-kernel SVM | `quantum_its/circuits.py`, `quantum_its/models.py` |
| Alternate-route recommendation | QUBO → Ising → QAOA (8 qubits) | `quantum_its/routing.py` |
| Reliability on noisy hardware | Depolarising-noise study, 3-qubit and Shor 9-qubit codes | `quantum_its/noise.py`, `quantum_its/qec.py` |
| Classical stages | Cleaning, features, scaling, baselines, decision, feasibility check | `quantum_its/data.py` |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU torch: --index-url https://download.pytorch.org/whl/cpu
# dataset (UCI #492): place Metro_Interstate_Traffic_Volume.csv in ../data/
python scripts/run_benchmarks.py           # every experiment -> results/results.json, figures/, models/ (~5 min)
python scripts/make_slides.py              # -> ../slides/QuantumITS.pptx
uvicorn app.server:app --port 8000         # dashboard at http://localhost:8000
```

## Dashboard
* **Overview**: the pipeline, which stages are classical and which are quantum, and what each symbol means.
* **Congestion prediction**: move the hour, day type, temperature and weather controls. You get the VQC decision, the
  probability from every model, its latency, and the measured 16-outcome distribution.
* **Route optimisation**: real demands for the chosen hour, solved by QAOA (p = 1–3), exact search and greedy search,
  plus the classical safety check on the emergency stream.
* **Error correction**: logical error rate of each code at any physical error rate (Qiskit Aer).
* **Results**: every benchmark from the paper.

## Design choices
* **Chronological split** (train 2012–16, test 2017–18), so no future hours leak into training.
* **Same inputs for every model.** Classical baselines get the hour as (sin, cos) so that they are not disadvantaged.
* **Verified simulation.** The PyTorch statevector simulator matches Qiskit to about 1e-15, and the QUBO matrix and Ising
  Hamiltonian are checked against the direct cost on all 256 plans (see `results.json`).
