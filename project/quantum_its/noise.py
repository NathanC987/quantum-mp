"""Run a trained VQC on Qiskit Aer with finite shots and depolarising gate noise.

Depolarising noise shrinks the measured <Z_0> towards 0 by a roughly constant factor f.
A one-parameter mitigation estimates f on a few training snapshots (least squares between noisy
and ideal <Z_0>) and divides it out before the classical decision.
"""
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from .circuits import to_qiskit


def noise_model(p2: float):
    """p2 = two-qubit (CNOT) error; single-qubit gates get p2/10, a typical hardware ratio."""
    nm = NoiseModel()
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["ry", "rz"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


def noisy_expval(model, X, p2=0.0, shots=1024, seed=0):
    theta, w = model.theta.numpy(), model.w.numpy()
    circuits = [to_qiskit(model.spec, x, theta, w if model.spec.scaling else None) for x in X]
    sim = AerSimulator(noise_model=noise_model(p2), seed_simulator=seed)
    result = sim.run(circuits, shots=shots).result()
    ev = []
    for i in range(len(circuits)):
        c = result.get_counts(i)
        ev.append((c.get("0", 0) - c.get("1", 0)) / shots)
    return np.array(ev)


def decide(model, ev, rescale=1.0):
    a, b = model.ab.numpy()
    return 1 / (1 + np.exp(-(a * ev / rescale + b)))


def fit_rescale(model, X_cal, p2, shots=1024):
    ideal = model.expval(X_cal).numpy()
    noisy = noisy_expval(model, X_cal, p2, shots, seed=99)
    return float(noisy @ ideal / (ideal @ ideal))
