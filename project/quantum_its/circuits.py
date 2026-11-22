"""Quantum stage: encoding U_encode(x), feature map U_phi(x), ansatz U(theta), measurement.

Circuits are simulated exactly with a small batched statevector simulator written in PyTorch
(one state per traffic snapshot, gradients by autograd). `to_qiskit` builds the identical circuit
in Qiskit for drawing, cross-checking and noisy (Aer) execution.

Qubit k carries feature x_k:  q0 = hour, q1 = day type, q2 = temperature, q3 = weather.
"""
import math

import numpy as np
import torch

N_QUBITS = 4
CDTYPE = torch.complex128


# ---------- batched statevector primitives: psi has shape (B, 2, 2, ..., 2), axis k+1 = qubit k ----------
def zero_state(batch: int, n: int = N_QUBITS) -> torch.Tensor:
    psi = torch.zeros((batch,) + (2,) * n, dtype=CDTYPE)
    psi[(slice(None),) + (0,) * n] = 1.0
    return psi


def _split(psi, q):
    return psi.select(q + 1, 0), psi.select(q + 1, 1)


def ry(psi, q, theta):
    a, b = _split(psi, q)
    c, s = _bcast(torch.cos(theta / 2), a), _bcast(torch.sin(theta / 2), a)
    return torch.stack([c * a - s * b, s * a + c * b], dim=q + 1)


def rz(psi, q, theta):
    a, b = _split(psi, q)
    ph = _bcast(torch.exp(-0.5j * theta.to(CDTYPE)), a)
    return torch.stack([ph * a, ph.conj() * b], dim=q + 1)


def cnot(psi, c, t):
    a, b = _split(psi, c)                                  # b = control is |1>
    t_ = t - 1 if t > c else t                             # target axis inside the sliced tensor
    return torch.stack([a, b.flip(t_ + 1)], dim=c + 1)


def zz_phase(psi, k, l, phi):
    """exp(-i * phi * Z_k Z_l): the feature-map unitary U_phi = exp(-i phi(x) H) with H = Z_k Z_l."""
    z = torch.tensor([1.0, -1.0], dtype=torch.float64)
    zk, zl = z.view(*[2 if i == k + 1 else 1 for i in range(psi.dim())]), \
        z.view(*[2 if i == l + 1 else 1 for i in range(psi.dim())])
    phase = torch.exp(-1j * _bcast(phi, psi).to(CDTYPE) * (zk * zl))
    return psi * phase


def expval_z(psi, q):
    p = psi.abs() ** 2
    a, b = p.select(q + 1, 0), p.select(q + 1, 1)
    return (a - b).reshape(psi.shape[0], -1).sum(1)


def probabilities(psi):
    return (psi.abs() ** 2).reshape(psi.shape[0], -1)


def _bcast(v, like):
    v = torch.as_tensor(v)
    if v.dim() == 0:
        return v
    return v.view(-1, *([1] * (like.dim() - 1)))


def ring_pairs(n=N_QUBITS):
    return [(k, (k + 1) % n) for k in range(n)]


def linear_pairs(n=N_QUBITS):
    return [(k, k + 1) for k in range(n - 1)]


def zz_coefficient(xk, xl):
    """phi_kl(x) = (pi - x_k)(pi - x_l): the standard second-order Pauli-ZZ feature-map coefficient."""
    return (math.pi - xk) * (math.pi - xl)


# ---------- circuit definition shared by the simulator and Qiskit ----------
class CircuitSpec:
    """Architecture of a variational quantum classifier.

    layers      L: number of (encode -> [feature map] -> trainable block) repetitions.
                L = 1 is the textbook VQC; L > 1 is data re-uploading.
    feature_map add the ZZ feature map U_phi(x) after each encoding.
    entangle    'ring' or 'linear' CNOT pattern inside the trainable block.
    scaling     trainable input scaling w * x_k inside the encoding rotations.
    ansatz_reps trainable rotation+entangle blocks per layer.
    """

    def __init__(self, layers=3, feature_map=False, entangle="ring", scaling=True, ansatz_reps=1):
        self.layers, self.feature_map, self.entangle = layers, feature_map, entangle
        self.scaling, self.ansatz_reps = scaling, ansatz_reps

    @property
    def pairs(self):
        return ring_pairs() if self.entangle == "ring" else linear_pairs()

    def n_params(self):
        return self.layers * self.ansatz_reps * N_QUBITS * 3 + (self.layers * N_QUBITS if self.scaling else 0) + 2

    def describe(self):
        return dict(layers=self.layers, feature_map=self.feature_map, entangle=self.entangle,
                    scaling=self.scaling, ansatz_reps=self.ansatz_reps, n_params=self.n_params())


def run_circuit(spec: CircuitSpec, X: torch.Tensor, theta: torch.Tensor, w: torch.Tensor | None):
    """Return the final state |Phi(x; theta)> for a batch of angle vectors X (B, 4)."""
    psi = zero_state(X.shape[0])
    for l in range(spec.layers):
        xs = X * w[l] if (spec.scaling and w is not None) else X
        for k in range(N_QUBITS):                                  # U_encode(x): RY(x_k)
            psi = ry(psi, k, xs[:, k])
        if spec.feature_map:                                       # U_phi(x): prod exp(-i phi_kl Z_k Z_l)
            for k, m in ring_pairs():
                psi = zz_phase(psi, k, m, zz_coefficient(X[:, k], X[:, m]))
        for r in range(spec.ansatz_reps):                          # U(theta): RZ RY RZ + CNOT entangler
            t = theta[l, r]
            for k in range(N_QUBITS):
                psi = rz(psi, k, t[k, 0]); psi = ry(psi, k, t[k, 1]); psi = rz(psi, k, t[k, 2])
            for c, tq in spec.pairs:
                psi = cnot(psi, c, tq)
    return psi


def to_qiskit(spec: CircuitSpec, x, theta, w=None, measure=True, symbolic=False):
    """The same circuit in Qiskit. symbolic=True uses Parameters (for labelled drawings)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    qc = QuantumCircuit(N_QUBITS, 1 if measure else 0)
    if symbolic:
        xv = ParameterVector("x", N_QUBITS)
        th = ParameterVector("θ", spec.layers * spec.ansatz_reps * N_QUBITS * 3)
        theta = np.array(list(th), dtype=object).reshape(spec.layers, spec.ansatz_reps, N_QUBITS, 3)
        wv = ParameterVector("w", spec.layers * N_QUBITS)
        w = np.array(list(wv), dtype=object).reshape(spec.layers, N_QUBITS) if spec.scaling else None
        x = list(xv)
    else:
        theta = np.asarray(theta)
        w = None if w is None else np.asarray(w)
    for l in range(spec.layers):
        for k in range(N_QUBITS):
            qc.ry(x[k] * w[l][k] if (spec.scaling and w is not None) else x[k], k)
        if spec.feature_map:
            for k, m in ring_pairs():
                qc.rzz(2 * (np.pi - x[k]) * (np.pi - x[m]), k, m)   # RZZ(2 phi) = exp(-i phi ZZ)
        for r in range(spec.ansatz_reps):
            for k in range(N_QUBITS):
                qc.rz(theta[l][r][k][0], k); qc.ry(theta[l][r][k][1], k); qc.rz(theta[l][r][k][2], k)
            for c, t in spec.pairs:
                qc.cx(c, t)
        if l < spec.layers - 1:
            qc.barrier()
    if measure:
        qc.measure(0, 0)
    return qc
