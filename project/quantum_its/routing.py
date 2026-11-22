"""Alternate-route recommendation as a QUBO, solved with QAOA and classical solvers.

N traffic streams approach the city. x_i = 1 diverts stream i to the alternate route.
Delay on each route grows quadratically with its load (a BPR-style congestion curve):

    J(x) = (D - d.x)^2 / C_main + (d.x)^2 / C_alt + tau * d.x + P * x_e

d = stream demands (veh/h, from real I-94 volumes), D = sum(d), tau = extra detour time per vehicle,
x_e = the stream carrying an emergency vehicle, which must stay on the (cleared) main route.
J is quadratic in binary x, i.e. a QUBO  J(x) = x^T Q x + const.
"""
import itertools
import time

import numpy as np
from scipy.optimize import minimize


def make_instance(demands, emergency=0, main_cap=0.55, alt_cap=0.35, tau=0.15, penalty=2.0):
    d = np.asarray(demands, float)
    D = d.sum()
    dn = d / D                                                  # work with load shares (scale-free)
    a = 1 / main_cap + 1 / alt_cap
    Q = a * np.outer(dn, dn)                                    # quadratic part
    lin = -2 * dn / main_cap + tau * dn                         # linear part
    lin[emergency] += penalty
    Q = np.triu(Q + Q.T) - np.diag(np.diag(Q))                  # upper-triangular, off-diagonal doubled
    Q[np.diag_indices_from(Q)] = np.diag(a * np.outer(dn, dn)) + lin   # x_i^2 = x_i
    const = 1 / main_cap
    return dict(Q=Q, const=const, demands=d, emergency=emergency, n=len(d),
                main_cap=main_cap, alt_cap=alt_cap, tau=tau, penalty=penalty)


def all_bitstrings(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)


def qubo_cost(inst, X):
    X = np.atleast_2d(X)
    return np.einsum("bi,ij,bj->b", X, inst["Q"], X) + inst["const"]


def direct_cost(inst, X):
    """J(x) computed straight from the traffic model (used to verify the QUBO matrix)."""
    X = np.atleast_2d(X)
    dn = inst["demands"] / inst["demands"].sum()
    alt = X @ dn
    return ((1 - alt) ** 2 / inst["main_cap"] + alt ** 2 / inst["alt_cap"] + inst["tau"] * alt
            + inst["penalty"] * X[:, inst["emergency"]])


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
