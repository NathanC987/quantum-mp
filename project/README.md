# QuantumITS — Hybrid Quantum–Classical Intelligent Transportation

Course project for **Case Study 5: National Intelligent Transportation System**. The platform has to predict
congestion, recommend alternate routes and do both within **one second**. This project tests where small
quantum circuits help, and keeps every other stage classical.

| Module | Quantum method | File |
|---|---|---|
| Congestion prediction | Data re-uploading variational quantum classifier (4 qubits, 50 parameters), quantum-kernel SVM | `quantum_its/circuits.py`, `quantum_its/models.py` |
| Detour planning, driven by the prediction | QUBO whose expected-delay cost is weighted by the classifier's P(busy) → Ising → QAOA (8 qubits, one per on-ramp) | `quantum_its/routing.py` |
| Reliability on noisy hardware | Device model with gate errors, T1/T2 decay (also while idle) and readout errors, on both quantum modules; readout correction + rescaling | `quantum_its/noise.py` |
| Classical stages | Cleaning, features, scaling, baselines, decision, feasibility check | `quantum_its/data.py` |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU torch: --index-url https://download.pytorch.org/whl/cpu
# dataset (UCI #492): place Metro_Interstate_Traffic_Volume.csv in ../data/
python scripts/run_benchmarks.py           # every experiment -> results/results.json, figures/, models/ (~10 min)
python scripts/run_benchmarks.py --deploy  # reuse the trained models; rerun noise, routing, end-to-end (~3 min)
python scripts/make_paper_assets.py        # -> ../paper/numbers.tex and figures
python scripts/make_slides.py              # -> ../slides/QuantumITS.pptx
uvicorn app.server:app --port 8000         # dashboard at http://localhost:8000
```

## Dashboard
Written for someone who knows nothing about ITS or the dataset. Every prediction is shown next to the real answer.
1. **The problem**: what ITS is, the I-94 dataset, the "busy hour" question (> 3,340 cars), the four clues, the fair
   train/test split, the average traffic by hour, a few raw rows, the headline result, and which stages are quantum.
2. **Check against real data**: the score of every model on all 15,246 test hours (2017–18), a right/wrong breakdown
   for the quantum model, and a day viewer. Pick any real day (or a random day, or a day with mistakes) to see each
   hour's real car count, the correct answer, the quantum and classical predictions with ✓/✗, the detour plan and the
   delay it really caused.
3. **Predict & plan**: one set of inputs (hour, day type, temperature, weather, ambulance ramp, QAOA depth, ideal or
   noisy hardware) drives both steps. Step 1 is every model's prediction. Step 2 is the detour plan built from the quantum
   model's P(busy), compared ramp by ramp with the best of all 256 plans. Below that is the ground truth: real 2017–18
   hours with the same conditions, and how much delay the plan would really have saved there.
4. **Hardware noise**: what is simulated and where, and its effect on the classifier, on P(busy) and on QAOA,
   with and without mitigation. It also explains why error correction is not used.
5. **All results**: every benchmark from the paper, each with a one-line explanation.

## How prediction and routing connect
Traffic joins the corridor at 8 on-ramps with fixed (assumed) shares; ramp i = qubit i, and x_i = 1 sends it on a detour.
The classifier gives P = P(busy). The plan minimises the expected delay
`J(x) = P·J_busy(x) + (1−P)·J_quiet(x) + penalty·x_ambulance`, where each J_state is a quadratic (BPR-style) delay on the
highway and the detour. Detours are released only when P > 0.5 (classical gate). Every plan is then scored with the real
volume and real state of that hour.

## Design choices
* **Chronological split** (train 2012–16, test 2017–18), so no future hours leak into training.
* **Same inputs for every model.** Classical baselines get the hour as (sin, cos) so that they are not disadvantaged.
* **Verified simulation.** The PyTorch statevector simulator matches Qiskit to about 1e-15, and the QUBO matrix and Ising
  Hamiltonian are checked against the direct cost on all 256 plans (see `results.json`).
