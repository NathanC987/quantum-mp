"""Detour planning driven by the congestion classifier, as a QUBO solved with QAOA and classical solvers.

Corridor model. Traffic reaches the I-94 corridor through N = 8 on-ramps ("streams"). Ramp i carries a fixed
share s_i of the corridor's traffic (an assumption: the dataset has a single sensor). x_i = 1 sends ramp i's
traffic on the detour, so the detour carries the share s.x and the highway the rest.

Delay in one traffic state (volume V in multiples of the busy threshold, highway capacity C):
    J_state(x) = (V (1 - s.x))^2 / C  +  (V s.x)^2 / C_alt  +  tau V s.x
(delay on each road grows with the square of its load, a quadratic BPR-style curve; tau = extra detour time).

Coupling to the classifier. The classifier gives P = P(busy) for the coming hour, and the plan minimises the
expected delay
    J(x) = P J_busy(x) + (1 - P) J_quiet(x)  +  pen x_e
with V_busy, V_quiet = mean volume of busy / not-busy training hours and C_busy < C_quiet (a busy highway is
near capacity). x_e is a ramp carrying an emergency vehicle, which must stay on the cleared highway.
J is quadratic in binary x, i.e. a QUBO  J(x) = x^T Q x + const.
"""
import itertools
import time

import numpy as np
from scipy.optimize import minimize

SHARES = np.array([0.20, 0.17, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07])   # share of corridor traffic per ramp (assumed)
C_BUSY, C_QUIET, C_ALT, TAU = 1.0, 3.0, 0.6, 0.8                      # capacities and detour time (threshold units)


def typical_volumes(train_volume, threshold):
    """V_busy, V_quiet: mean volume of busy and not-busy training hours, in multiples of the threshold."""
    v = np.asarray(train_volume, float) / threshold
    return float(v[v > 1].mean()), float(v[v <= 1].mean())


def state_cost(X, volume, capacity, shares=SHARES):
    """J_state for plans X (rows of 0/1), straight from the traffic model."""
    a = volume * (np.atleast_2d(X) @ shares)
    return (volume - a) ** 2 / capacity + a ** 2 / C_ALT + TAU * a


def best_plans(P, v_busy, v_quiet, gate=True):
    """Exact best plan (a row of all_bitstrings(8)) for every P(busy), vectorised over many hours.
    gate: the classical decision layer releases detours only when P > 0.5."""
    X = all_bitstrings(len(SHARES))
    Jb, Jq = state_cost(X, v_busy, C_BUSY), state_cost(X, v_quiet, C_QUIET)
    P = np.atleast_1d(np.asarray(P, float))
    idx = np.argmin(P[:, None] * Jb[None] + (1 - P[:, None]) * Jq[None], axis=1)
    if gate:
        idx[P <= 0.5] = 0
    return X[idx]


def realised_all(volume, busy):
    """Delay of every one of the 256 plans for each hour, given the real volume (threshold units) and state."""
    X = all_bitstrings(len(SHARES))
    V = np.atleast_1d(np.asarray(volume, float))[:, None]
    cap = np.where(np.atleast_1d(busy), C_BUSY, C_QUIET)[:, None]
    a = V * (X @ SHARES)[None]
    return (V - a) ** 2 / cap + a ** 2 / C_ALT + TAU * a


def make_instance(p_busy, v_busy, v_quiet, emergency=None, shares=SHARES):
    s = np.asarray(shares, float)
    P = float(p_busy)
    terms = [(P, v_busy, C_BUSY), (1 - P, v_quiet, C_QUIET)]
    const = sum(w * V ** 2 / C for w, V, C in terms)                     # J(no detour)
    A = sum(w * V ** 2 * (1 / C + 1 / C_ALT) for w, V, C in terms)       # coefficient of (s.x)^2
    L = sum(w * (-2 * V ** 2 / C + TAU * V) for w, V, C in terms)        # coefficient of s.x
    Q = A * np.outer(s, s)
    Q = np.triu(Q + Q.T) - np.diag(np.diag(Q))                           # upper-triangular, off-diagonal doubled
    Q[np.diag_indices_from(Q)] = A * s ** 2 + L * s                      # x_i^2 = x_i
    penalty = 1.0 + 2 * const                                            # larger than any possible saving
    if emergency is not None:
        Q[emergency, emergency] += penalty
    return dict(Q=Q, const=const, shares=s, n=len(s), p_busy=P, v_busy=v_busy, v_quiet=v_quiet,
                emergency=emergency, penalty=penalty)


def all_bitstrings(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)


def qubo_cost(inst, X):
    X = np.atleast_2d(X)
    return np.einsum("bi,ij,bj->b", X, inst["Q"], X) + inst["const"]


def direct_cost(inst, X):
    """J(x) computed straight from the traffic model (used to verify the QUBO matrix)."""
    X = np.atleast_2d(X); P = inst["p_busy"]
    J = (P * state_cost(X, inst["v_busy"], C_BUSY, inst["shares"])
         + (1 - P) * state_cost(X, inst["v_quiet"], C_QUIET, inst["shares"]))
    if inst["emergency"] is not None:
        J = J + inst["penalty"] * X[:, inst["emergency"]]
    return J


def realised_cost(X, volume, busy, shares=SHARES):
    """Delay a plan really causes, given the hour's real volume (threshold units) and real state."""
    return state_cost(X, volume, C_BUSY if busy else C_QUIET, shares)


def ising(inst):
    """QUBO -> Ising with x_i = (1 - Z_i)/2:  H_C = sum h_i Z_i + sum J_ij Z_i Z_j + offset."""
    Q, n = inst["Q"], inst["n"]
    h, J, off = np.zeros(n), np.zeros((n, n)), inst["const"]
    for i in range(n):
        h[i] -= Q[i, i] / 2; off += Q[i, i] / 2
        for j in range(i + 1, n):
            q = Q[i, j] / 4
            J[i, j] += q; h[i] -= q; h[j] -= q; off += q
    return h, J, off


# ---------------- QAOA (exact statevector, 2^n amplitudes) ----------------
def _mixer(psi, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape((2,) * n)
    for q in range(n):
        a, b = np.take(psi, 0, axis=q), np.take(psi, 1, axis=q)
        psi = np.stack([c * a + s * b, s * a + c * b], axis=q)
    return psi.reshape(-1)


def qaoa_state(costs, gammas, betas, n):
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)          # H^n |0>: uniform superposition
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * costs)                     # U_C(gamma) = exp(-i gamma H_C)
        psi = _mixer(psi, b, n)                                 # U_M(beta)  = exp(-i beta sum X_i)
    return psi


def _bit_order(n):
    # amplitude index -> bitstring with qubit q = bit q (matches the reshape in _mixer: axis 0 = MSB)
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(float)


def run_qaoa(inst, p=1, restarts=4, seed=0):
    n = inst["n"]
    X = _bit_order(n)
    costs = qubo_cost(inst, X)
    scale = costs.max() - costs.min()
    c = (costs - costs.min()) / scale                           # normalised cost Hamiltonian diagonal
    energy = lambda th: float(np.abs(qaoa_state(c, th[:p], th[p:], n)) ** 2 @ c)
    rng = np.random.default_rng(seed)
    best = None
    t0 = time.perf_counter()
    for r in range(restarts):
        # linear-ramp start (annealing-inspired) for the first restart, random for the others
        x0 = (np.concatenate([np.linspace(0.1, 1, p) * 1.5, np.linspace(1, 0.1, p) * 0.6]) if r == 0
              else rng.uniform(0, np.pi, 2 * p))
        res = minimize(energy, x0, method="COBYLA", options={"maxiter": 300})
        if best is None or res.fun < best.fun:
            best = res
    seconds = time.perf_counter() - t0
    probs = np.abs(qaoa_state(c, best.x[:p], best.x[p:], n)) ** 2
    opt = costs.min()
    return dict(p=p, params=best.x, probs=probs, X=X, costs=costs, seconds=seconds,
                approx_ratio=1 - float(probs @ c),              # 1 = all probability on the optimum
                p_opt=float(probs[np.isclose(costs, opt)].sum()),
                p_near=float(probs[c <= 0.02].sum()),                # plans within 2% of the optimum
                best_sample=X[np.argmax(probs)], n_evals=best.nfev)


def qaoa_circuit(inst, gammas, betas, measure=True):
    """Qiskit version of qaoa_state, with the cost normalised exactly as in run_qaoa. Qubit i = ramp i."""
    from qiskit import QuantumCircuit
    n = inst["n"]
    costs = qubo_cost(inst, all_bitstrings(n)); scale = costs.max() - costs.min()
    h, J, _ = ising(inst)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i in range(n):
            qc.rz(2 * g * h[i] / scale, i)                              # exp(-i g h_i Z_i / scale)
        for i, j in itertools.combinations(range(n), 2):
            if abs(J[i, j]) > 1e-12:
                qc.rzz(2 * g * J[i, j] / scale, i, j)                   # exp(-i g J_ij Z_i Z_j / scale)
        qc.rx(2 * b, range(n))                                          # mixer exp(-i b X)
    if measure:
        qc.measure_all()
    return qc


def counts_to_plans(counts, n):
    """Qiskit counts -> (plans X, probabilities). Qiskit keys are little-endian: the last character is qubit 0."""
    keys = list(counts)
    X = np.array([[int(k.replace(" ", "")[n - 1 - q]) for q in range(n)] for k in keys], float)
    w = np.array([counts[k] for k in keys], float)
    return X, w / w.sum()


def sample_plan(qaoa, shots=256, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.choice(len(qaoa["probs"]), size=shots, p=qaoa["probs"] / qaoa["probs"].sum())
    best = s[np.argmin(qaoa["costs"][s])]
    return qaoa["X"][best], qaoa["costs"][best]


# ---------------- classical solvers ----------------
def solve_exact(inst):
    t0 = time.perf_counter()
    X = all_bitstrings(inst["n"]); c = qubo_cost(inst, X); i = int(np.argmin(c))
    return X[i], float(c[i]), time.perf_counter() - t0


def solve_greedy(inst):
    t0 = time.perf_counter()
    x = np.zeros(inst["n"]); cur = qubo_cost(inst, x)[0]
    while True:
        trials = [(qubo_cost(inst, np.where(np.arange(inst["n"]) == i, 1 - x, x))[0], i) for i in range(inst["n"])]
        c, i = min(trials)
        if c >= cur - 1e-12:
            break
        x[i] = 1 - x[i]; cur = c
    return x, float(cur), time.perf_counter() - t0


def solve_annealing(inst, steps=2000, seed=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, inst["n"]).astype(float); cur = qubo_cost(inst, x)[0]
    best_x, best = x.copy(), cur
    for k in range(steps):
        T = 0.5 * (1 - k / steps) + 1e-3
        i = rng.integers(inst["n"]); y = x.copy(); y[i] = 1 - y[i]
        c = qubo_cost(inst, y)[0]
        if c < cur or rng.random() < np.exp(-(c - cur) / T):
            x, cur = y, c
            if c < best:
                best_x, best = y.copy(), c
    return best_x, float(best), time.perf_counter() - t0
