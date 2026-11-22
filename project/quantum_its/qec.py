"""Quantum error correction for one traffic qubit: 3-qubit bit/phase-flip codes and Shor's 9-qubit code.

Same constructions as Experiments 7 and 8. The logical error rate is estimated with three test
states |0>, |+>, |+i>: each logical Pauli error (X, Y or Z) is revealed by exactly two of them,
so p_L = (f_0 + f_+ + f_+i) / 2.
"""
from functools import lru_cache

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

BLOCKS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]


def body_unprotected(qc):
    qc.id(0)


def body_bit(qc):
    qc.cx(0, 1); qc.cx(0, 2); qc.id(range(3)); qc.cx(0, 1); qc.cx(0, 2); qc.ccx(2, 1, 0)


def body_phase(qc):
    qc.cx(0, 1); qc.cx(0, 2); qc.h(range(3)); qc.id(range(3)); qc.h(range(3))
    qc.cx(0, 1); qc.cx(0, 2); qc.ccx(2, 1, 0)


def shor_encode(qc):
    qc.cx(0, 3); qc.cx(0, 6); qc.h([0, 3, 6])
    for a, b, c in BLOCKS:
        qc.cx(a, b); qc.cx(a, c)


def shor_decode_correct(qc):
    for a, b, c in BLOCKS:
        qc.cx(a, b); qc.cx(a, c); qc.ccx(c, b, a)
    qc.h([0, 3, 6]); qc.cx(0, 3); qc.cx(0, 6); qc.ccx(6, 3, 0)


def body_shor(qc):
    shor_encode(qc); qc.id(range(9)); shor_decode_correct(qc)


CODES = {"unprotected": (body_unprotected, 1), "bit_flip": (body_bit, 3),
         "phase_flip": (body_phase, 3), "shor": (body_shor, 9)}
PREP = {"0": [], "+": ["h"], "+i": ["h", "s"]}


def _test_circuit(body, n, state):
    qc = QuantumCircuit(n, 1)
    for g in PREP[state]:
        getattr(qc, g)(0)
    body(qc)
    for g in reversed(PREP[state]):
        getattr(qc, "sdg" if g == "s" else g)(0)
    qc.measure(0, 0)
    return qc


@lru_cache(maxsize=512)
def logical_error_rate(code: str, p: float, noise: str = "depolarizing", shots: int = 8000, seed: int = 1):
    body, n = CODES[code]
    if noise == "depolarizing":
        err = depolarizing_error(4 * p / 3, 1)                  # X, Y, Z each with probability p/3
    else:
        err = pauli_error([("X" if noise == "bit" else "Z", p), ("I", 1 - p)])
    nm = NoiseModel(); nm.add_all_qubit_quantum_error(err, ["id"])
    sim = AerSimulator(noise_model=nm, seed_simulator=seed)
    f = [sim.run(_test_circuit(body, n, s), shots=shots).result().get_counts().get("1", 0) / shots
         for s in PREP]
    return 0.5 * sum(f)


def physical_qubits(code: str) -> int:
    return CODES[code][1]
