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
Written for someone who knows nothing about ITS or the dataset. Every prediction is shown next to the real answer.
1. **The problem**: what ITS is, the I-94 dataset, the "busy hour" question (> 3,340 cars), the four clues, the fair
   train/test split, the average traffic by hour, a few raw rows, the headline result, and which stages are quantum.
2. **Check against real data**: the score of every model on all 15,246 test hours (2017–18), a right/wrong breakdown
   for the quantum model, and a day viewer. Pick any real day (or a random day, or a day with mistakes) to see each
   hour's real car count, the correct answer, and the quantum and classical predictions with ✓/✗.
3. **Try it yourself**: set the hour, day type, temperature and weather to get every model's prediction, then compare
   with the real 2017–18 hours that had the same conditions. The circuit and its 16-outcome readout are in a fold-out.
4. **Detour planner**: QAOA plan vs the best of all 256 plans (found by exhaustive search), compared stream by stream,
   with its rank and the ambulance safety check.
5. **Error correction**: how often a stored value is read back wrong, with and without each code (Qiskit Aer).
6. **All results**: every benchmark from the paper, each with a one-line explanation.

## Design choices
* **Chronological split** (train 2012–16, test 2017–18), so no future hours leak into training.
* **Same inputs for every model.** Classical baselines get the hour as (sin, cos) so that they are not disadvantaged.
* **Verified simulation.** The PyTorch statevector simulator matches Qiskit to about 1e-15, and the QUBO matrix and Ising
  Hamiltonian are checked against the direct cost on all 256 plans (see `results.json`).
