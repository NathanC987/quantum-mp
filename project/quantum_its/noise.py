"""Hardware noise: run circuits on a simulated noisy device, and mitigate the errors.

The device model has every main error source of a superconducting chip:
  * gate errors       depolarising error on every 1-qubit (sx, x) and 2-qubit (cx) gate
  * decoherence       T1/T2 thermal relaxation during every gate and while a qubit waits (idle delays,
                      inserted by scheduling the transpiled circuit)
  * readout errors    a measured 0 is read as 1 (and vice versa) with probability err_ro
The qubits are connected in a ring, so gates between distant qubits need extra SWAPs (as on hardware).

`level` scales the whole profile: level 1 = PROFILE (typical of current superconducting devices),
level 2 = every error rate doubled and T1, T2 halved, level 0 = ideal (shot noise only).

Mitigation (no extra qubits, unlike error correction):
  1. readout: measure the 2x2 confusion matrix of the read-out qubit once and invert it;
  2. rescaling: gate noise and decoherence shrink <Z_0> towards 0 by a roughly constant factor f;
     estimate f on a few training snapshots and divide it out before the classical decision.
"""
import warnings
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap, InstructionProperties, QubitProperties
from qiskit_aer import AerSimulator

from .circuits import to_qiskit

warnings.filterwarnings("ignore", message="Duration is rounded")   # durations snapped to the device clock

PROFILE = dict(err_1q=2.5e-4, err_2q=7.5e-3, err_ro=1.5e-2,      # error probabilities
               t1=150e-6, t2=100e-6,                              # seconds
               t_1q=36e-9, t_2q=500e-9, t_ro=1.2e-6)              # gate and read-out durations (seconds)


@lru_cache(maxsize=32)
def device(n: int, level: float):
    """A simulated n-qubit ring-connected device with the noise PROFILE scaled by `level`."""
    b = GenericBackendV2(n, basis_gates=["cx", "id", "rz", "sx", "x"], coupling_map=CouplingMap.from_ring(n), seed=1)
    t, P = b.target, PROFILE
    for name in ("sx", "x", "id"):
        for q in range(n):
            t.update_instruction_properties(name, (q,), InstructionProperties(duration=P["t_1q"], error=level * P["err_1q"]))
    for q in range(n):
        t.update_instruction_properties("rz", (q,), InstructionProperties(duration=0.0, error=0.0))   # virtual gate
        t.update_instruction_properties("measure", (q,), InstructionProperties(duration=P["t_ro"], error=level * P["err_ro"]))
    t.qubit_properties = [QubitProperties(t1=P["t1"] / level, t2=P["t2"] / level, frequency=5e9)] * n
    for qargs in t["cx"]:
        t.update_instruction_properties("cx", qargs, InstructionProperties(duration=P["t_2q"], error=level * P["err_2q"]))
    return b, AerSimulator.from_backend(b)


def run(circuits, n, level, shots=1024, seed=0):
    """Counts for each circuit. level 0 = ideal simulator (finite shots only)."""
    if level == 0:
        res = AerSimulator(seed_simulator=seed).run(circuits, shots=shots).result()
        return [res.get_counts(i) for i in range(len(circuits))]
    backend, sim = device(n, float(level))
    tc = transpile(circuits, backend=backend, initial_layout=list(range(n)), scheduling_method="alap",
                   optimization_level=1, seed_transpiler=0)
    res = sim.run(tc, shots=shots, seed_simulator=seed).result()
    return [res.get_counts(i) for i in range(len(circuits))]


def readout_matrix(level, n=4, shots=8192):
    """M[i, j] = P(read i | prepared j) for qubit 0, measured on the device like any other circuit."""
    preps = []
    for j in (0, 1):
        qc = QuantumCircuit(n, 1)
        if j:
            qc.x(0)
        qc.measure(0, 0); preps.append(qc)
    c = run(preps, n, level, shots, seed=7)
    p1 = np.array([c[j].get("1", 0) / shots for j in (0, 1)])
    return np.array([1 - p1, p1])


def noisy_expval(model, X, level, shots=1024, seed=0, M=None):
    """Measured <Z_0> of the VQC on the noisy device; with M given, readout errors are inverted first."""
    theta, w = model.theta.numpy(), model.w.numpy()
    circuits = [to_qiskit(model.spec, x, theta, w if model.spec.scaling else None) for x in X]
    p1 = np.array([c.get("1", 0) / shots for c in run(circuits, 4, level, shots, seed)])
    if M is not None:
        p1 = np.clip((p1 - M[1, 0]) / (M[1, 1] - M[1, 0]), 0, 1)
    return 1 - 2 * p1


def decide(model, ev, rescale=1.0):
    a, b = model.ab.numpy()
    return 1 / (1 + np.exp(-(a * ev / rescale + b)))


def fit_rescale(model, X_cal, level, M=None, shots=1024):
    """Least-squares shrink factor f between noisy (readout-corrected) and ideal <Z_0> on calibration snapshots."""
    ideal = model.expval(X_cal).numpy()
    noisy = noisy_expval(model, X_cal, level, shots, seed=99, M=M)
    return float(noisy @ ideal / (ideal @ ideal))
