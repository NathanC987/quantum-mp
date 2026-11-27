"""Runs every experiment in the paper and writes results/results.json, figures/*.png, models/*.

    python scripts/run_benchmarks.py            # full run (~10 min on a laptop CPU)
    python scripts/run_benchmarks.py --quick    # smoke test
    python scripts/run_benchmarks.py --deploy   # reuse the trained models, rerun only sections 4-7
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantum_its import data as D, noise, routing  # noqa: E402
from quantum_its.circuits import CircuitSpec, to_qiskit  # noqa: E402
from quantum_its.models import (VQC, QuantumKernelSVM, classical_inputs, classical_models,  # noqa: E402
                                metrics)
from quantum_its.noise import decide  # noqa: E402

FIG, RES, MOD = ROOT / "figures", ROOT / "results", ROOT / "models"
for p in (FIG, RES, MOD):
    p.mkdir(exist_ok=True)

PROPOSED = dict(layers=3, feature_map=False, entangle="ring", scaling=True, ansatz_reps=1)
BASELINE = dict(layers=1, feature_map=False, entangle="linear", scaling=False, ansatz_reps=2)   # Exp 5 style
COLORS = {"quantum": "#7c3aed", "classical": "#0ea5e9"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def train_vqc(spec_kw, X, y, seed, epochs=150):
    return VQC(CircuitSpec(**spec_kw), seed=seed).fit(X, y, epochs=epochs)


def fit_all(Xtr, ytr, seed):
    """Every model in the comparison, trained on the same snapshots."""
    out = {}
    t = time.perf_counter(); m = train_vqc(PROPOSED, Xtr, ytr, seed)
    out["Re-uploading VQC (ours)"] = ("quantum", m, m.predict_proba, time.perf_counter() - t)
    t = time.perf_counter(); m = train_vqc(BASELINE, Xtr, ytr, seed)
    out["Standard VQC"] = ("quantum", m, m.predict_proba, time.perf_counter() - t)
    t = time.perf_counter(); m = QuantumKernelSVM().fit(Xtr, ytr)
    out["Quantum-kernel SVM"] = ("quantum", m, m.predict_proba, time.perf_counter() - t)
    for name, make in classical_models().items():
        t = time.perf_counter(); m = make().fit(classical_inputs(Xtr), ytr)
        out[name] = ("classical", m, (lambda X, m=m: m.predict_proba(classical_inputs(X))[:, 1]),
                     time.perf_counter() - t)
    return out


def main(quick=False, deploy=False):
    seeds = [0] if quick else [0, 1, 2]
    sizes = [50, 200] if quick else [50, 100, 250, 500, 1000, 2000]
    data = D.load_dataset(n_test=500 if quick else 2000)
    Xte, yte = data["X_test"], data["y_test"]
    R = dict(dataset=dict(n_rows=len(data["df"]), n_train_pool=data["n_pool"], n_test_period=data["n_test_total"],
                          n_test_used=len(yte), threshold=data["threshold"], test_positive_rate=float(yte.mean()),
                          features=D.FEATURES))
    log("dataset", R["dataset"])
    if deploy:                                                  # reuse sections 0-3 and the saved models
        old = json.loads((RES / "results.json").read_text())
        R.update({k: v for k, v in old.items() if k in ("check_sim_vs_qiskit_max_abs_diff", "comparison", "ablation", "final")})
        final = VQC.from_state({**dict(np.load(MOD / "vqc.npz")), "spec": json.loads((MOD / "vqc_spec.json").read_text())})
        with open(MOD / "classical.pkl", "rb") as f:
            saved = pickle.load(f)
        clf, qksvm = saved["classical"], saved["qksvm"]
        Xtr, ytr = D.sample_train(data, 200 if quick else 1000, 0)
    else:
        final, clf, qksvm, Xtr = train_and_compare(R, data, quick, seeds, sizes)
    deployment(R, data, final, clf, qksvm, Xtr, quick)
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=float))
    make_figures(R, final)
    log("all done")


def train_and_compare(R, data, quick, seeds, sizes):
    Xte, yte = data["X_test"], data["y_test"]
    # ---------------- 0. correctness checks ----------------
    from qiskit.quantum_info import SparsePauliOp, Statevector
    spec = CircuitSpec(**PROPOSED); m = VQC(spec, seed=5); m.w = torch.rand(m.w.shape) + 0.5
    ev = m.expval(Xte[:20]).numpy()
    qk = [Statevector(to_qiskit(spec, x, m.theta.numpy(), m.w.numpy(), measure=False))
          .expectation_value(SparsePauliOp("IIIZ")).real for x in Xte[:20]]
    R["check_sim_vs_qiskit_max_abs_diff"] = float(np.abs(ev - np.array(qk)).max())
    log("simulator vs qiskit max diff", R["check_sim_vs_qiskit_max_abs_diff"])

    # ---------------- 1. model comparison + learning curve ----------------
    rows = []
    for n in sizes:
        for s in seeds:
            Xtr, ytr = D.sample_train(data, n, s)
            for name, (fam, model, proba, fit_s) in fit_all(Xtr, ytr, s).items():
                rows.append(dict(model=name, family=fam, n=n, seed=s, fit_s=fit_s, **metrics(yte, proba(Xte))))
            log(f"n={n} seed={s} done")
    R["comparison"] = rows

    # ---------------- 2. ablation of the VQC design (n = 500) ----------------
    n_ab = 200 if quick else 500
    variants = {
        "Standard VQC (encode once, linear)": BASELINE,
        "+ ZZ feature map": dict(BASELINE, feature_map=True),
        "Re-uploading L=3, ring, scaling (ours)": PROPOSED,
        "  without input scaling": dict(PROPOSED, scaling=False),
        "  with linear entanglement": dict(PROPOSED, entangle="linear"),
        "  with ZZ feature map": dict(PROPOSED, feature_map=True),
        "  L=1": dict(PROPOSED, layers=1), "  L=2": dict(PROPOSED, layers=2),
        "  L=4": dict(PROPOSED, layers=4), "  L=5": dict(PROPOSED, layers=5),
    }
    abl = []
    for name, kw in variants.items():
        for s in seeds:
            Xtr, ytr = D.sample_train(data, n_ab, s)
            mdl = train_vqc(kw, Xtr, ytr, s)
            abl.append(dict(variant=name, seed=s, n_params=CircuitSpec(**kw).n_params(),
                            final_loss=mdl.history[-1], **metrics(yte, mdl.predict_proba(Xte))))
    # linear (non-cyclic) hour encoding: hour -> [0, pi] instead of [0, 2 pi)
    for s in seeds:
        Xtr, ytr = D.sample_train(data, n_ab, s)
        lin = lambda X: np.column_stack([X[:, 0] / 2, X[:, 1:]])          # 2*pi*h/24 -> pi*h/24
        mdl = train_vqc(PROPOSED, lin(Xtr), ytr, s)
        abl.append(dict(variant="  with linear (non-cyclic) hour encoding", seed=s,
                        n_params=CircuitSpec(**PROPOSED).n_params(), final_loss=mdl.history[-1],
                        **metrics(yte, mdl.predict_proba(lin(Xte)))))
    R["ablation"] = abl
    log("ablation done")

    # ---------------- 3. final deployed models (n = 1000) ----------------
    Xtr, ytr = D.sample_train(data, 200 if quick else 1000, 0)
    final = train_vqc(PROPOSED, Xtr, ytr, 0, epochs=200)
    np.savez(MOD / "vqc.npz", **{k: v for k, v in final.state_dict().items() if k != "spec"})
    (MOD / "vqc_spec.json").write_text(json.dumps(final.spec.describe()))
    clf = {name: make().fit(classical_inputs(Xtr), ytr) for name, make in classical_models().items()}
    qksvm = QuantumKernelSVM().fit(Xtr[:500], ytr[:500])
    with open(MOD / "classical.pkl", "wb") as f:
        pickle.dump(dict(classical=clf, qksvm=qksvm), f)
    R["final"] = dict(vqc=metrics(yte, final.predict_proba(Xte)), loss=final.history,
                      vqc_spec=final.spec.describe(),
                      rf=metrics(yte, clf["Random forest"].predict_proba(classical_inputs(Xte))[:, 1]))
    log("final model", R["final"]["vqc"])
    return final, clf, qksvm, Xtr


def deployment(R, data, final, clf, qksvm, Xtr, quick):
    Xte, yte, vte = data["X_test"], data["y_test"], data["v_test"]
    thr = data["threshold"]

    # ---------------- 4. hardware noise and mitigation ----------------
    # Device model: gate errors + T1/T2 decoherence (also while idle) + readout errors, scaled by `level`.
    k = 150 if quick else 500
    Xn, yn, vn = Xte[:k], yte[:k], vte[:k]
    Xcal = Xtr[:100]
    p_ideal = final.predict_proba(Xn)                          # exact probabilities: no noise, no shots
    R["noise_ideal_acc"] = metrics(yn, p_ideal)["acc"]
    R["noise_profile"] = noise.PROFILE
    noise_rows, noisy_P = [], {}
    for lv in ([0, 1] if quick else [0, 0.5, 1, 2, 4]):
        M = noise.readout_matrix(lv) if lv else None
        ev = noise.noisy_expval(final, Xn, lv)
        evm = noise.noisy_expval(final, Xn, lv, M=M) if lv else ev
        f = noise.fit_rescale(final, Xcal, lv, M=M) if lv else 1.0
        P_raw, P_ro, P_mit = decide(final, ev), decide(final, evm), decide(final, evm, f)
        noisy_P[lv] = (P_raw, P_mit)
        noise_rows.append(dict(level=lv, readout_err=float(0 if M is None else (M[1, 0] + M[0, 1]) / 2), shrink=f,
                               acc_raw=metrics(yn, P_raw)["acc"], acc_mitigated=metrics(yn, P_mit)["acc"],
                               prob_err_raw=float(np.abs(P_raw - p_ideal).mean()),
                               prob_err_readout=float(np.abs(P_ro - p_ideal).mean()),
                               prob_err_mitigated=float(np.abs(P_mit - p_ideal).mean())))
        log("noise", noise_rows[-1])
    R["noise"] = noise_rows
    shot_rows = []
    for shots in ([64, 1024] if quick else [16, 64, 256, 1024, 4096]):
        ev = noise.noisy_expval(final, Xn, 0, shots=shots)
        shot_rows.append(dict(shots=shots, acc=metrics(yn, decide(final, ev))["acc"]))
    R["shots"] = shot_rows

    # ---------------- 5. latency vs the one-second budget ----------------
    from qiskit_aer import AerSimulator
    def timeit(fn, reps=100):
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
        return dict(median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)))
    x1 = Xte[:1]
    aer = AerSimulator()
    circ = to_qiskit(final.spec, x1[0], final.theta.numpy(), final.w.numpy())
    VB, VQ = routing.typical_volumes(data["train_volume"], thr)
    lat = {
        "Preprocessing (classical)": timeit(lambda: D.to_angles(17, 0, 290.0, 1)),
        "VQC – exact simulator": timeit(lambda: final.predict_proba(x1)),
        "VQC – Qiskit Aer, 1024 shots": timeit(lambda: aer.run(to_qiskit(final.spec, x1[0], final.theta.numpy(),
                                                                         final.w.numpy()), shots=1024).result(), 30),
        "VQC – noisy device, 1024 shots": timeit(lambda: noise.noisy_expval(final, x1, 1), 20),
        "Quantum-kernel SVM": timeit(lambda: qksvm.predict_proba(x1), 30),
        "Random forest": timeit(lambda: clf["Random forest"].predict_proba(classical_inputs(x1)), 30),
        "MLP": timeit(lambda: clf["MLP (16-16)"].predict_proba(classical_inputs(x1))),
        "Detour plan – exact (classical)": timeit(lambda: routing.solve_exact(routing.make_instance(0.9, VB, VQ)), 30),
        "Detour plan – QAOA p=1 (simulated)": timeit(lambda: routing.run_qaoa(routing.make_instance(0.9, VB, VQ), p=1, restarts=1), 5),
    }
    R["latency"] = lat
    R["circuit"] = dict(depth=circ.decompose().depth(), cx=circ.count_ops().get("cx", 0),
                        n_params=final.spec.n_params(), qubits=4)
    log("latency", {k: round(v["median_ms"], 3) for k, v in lat.items()})

    # ---------------- 6. detour planning: QUBO built from the classifier's P(busy) ----------------
    R["routing"] = dict(v_busy=VB, v_quiet=VQ, shares=routing.SHARES.tolist(), c_busy=routing.C_BUSY,
                        c_quiet=routing.C_QUIET, c_alt=routing.C_ALT, tau=routing.TAU)
    rng = np.random.default_rng(0)
    P_te = final.predict_proba(Xte)
    busy_idx = rng.choice(np.flatnonzero(P_te > 0.5), 5 if quick else 30, replace=False)
    emerg = [None if e < 0 else int(e) for e in rng.integers(-1, 8, len(busy_idx))]
    insts = [routing.make_instance(P_te[i], VB, VQ, emergency=e) for i, e in zip(busy_idx, emerg)]
    X8 = routing.all_bitstrings(8)
    R["qubo_check_max_abs_diff"] = float(max(np.abs(routing.qubo_cost(i, X8) - routing.direct_cost(i, X8)).max()
                                             for i in insts))
    h, J, off = routing.ising(insts[0]); z = 1 - 2 * X8
    R["ising_check_max_abs_diff"] = float(np.abs(z @ h + np.einsum("bi,ij,bj->b", z, J, z) + off
                                                 - routing.qubo_cost(insts[0], X8)).max())
    log("QUBO / Ising checks", R["qubo_check_max_abs_diff"], R["ising_check_max_abs_diff"])
    qrows, qaoa_p2 = [], []
    for k, inst in enumerate(insts):
        xopt, copt, t_ex = routing.solve_exact(inst)
        cmax = routing.qubo_cost(inst, X8).max()
        gap = lambda c: float((c - copt) / (cmax - copt))
        _, cg, tg = routing.solve_greedy(inst); _, ca, ta = routing.solve_annealing(inst, seed=k)
        qrows += [dict(inst=k, method="Exact enumeration", gap=0.0, ms=t_ex * 1000),
                  dict(inst=k, method="Greedy", gap=gap(cg), ms=tg * 1000),
                  dict(inst=k, method="Simulated annealing", gap=gap(ca), ms=ta * 1000)]
        cn = (routing.qubo_cost(inst, X8) - copt) / (cmax - copt)
        qrows.append(dict(inst=k, method="Random guess (p=0)", approx_ratio=float(1 - cn.mean()),
                          p_opt=float(np.isclose(cn, 0).mean()), p_near=float((cn <= 0.02).mean())))
        for p in ([1, 2] if quick else [1, 2, 3, 4]):
            q = routing.run_qaoa(inst, p=p, restarts=3, seed=k)
            _, cs = routing.sample_plan(q, shots=256, seed=k)
            qrows.append(dict(inst=k, method=f"QAOA p={p}", approx_ratio=q["approx_ratio"], p_opt=q["p_opt"], p_near=q["p_near"],
                              gap=gap(cs), ms=q["seconds"] * 1000, evals=q["n_evals"]))
            if p == 2:
                qaoa_p2.append(q)
    R["qaoa"] = qrows
    log("QAOA done")

    # QAOA on the noisy device (parameters trained noise-free, then run as on hardware; best of 256 shots kept)
    qn_rows = []
    for k, (inst, q) in enumerate(list(zip(insts, qaoa_p2))[: 3 if quick else 15]):
        circ_q = routing.qaoa_circuit(inst, q["params"][:2], q["params"][2:])
        costs = routing.qubo_cost(inst, X8); cmin, cmax = costs.min(), costs.max()
        for lv in ([0, 1] if quick else [0, 0.5, 1, 2, 4]):
            c = noise.run([circ_q], 8, lv, shots=256, seed=k)[0]
            Xs, w = routing.counts_to_plans(c, 8)
            cs = routing.qubo_cost(inst, Xs); cn = (cs - cmin) / (cmax - cmin)
            qn_rows.append(dict(inst=k, level=lv, approx_ratio=float(1 - w @ cn), p_opt=float(w[np.isclose(cn, 0)].sum()),
                                best_gap=float(cn.min())))
    R["qaoa_noise"] = qn_rows
    if not quick:
        from qiskit import transpile
        tq = transpile(routing.qaoa_circuit(insts[0], [0.5, 0.4], [0.6, 0.3]),
                       backend=noise.device(8, 1.0)[0], initial_layout=list(range(8)), optimization_level=1, seed_transpiler=0)
        R["qaoa_circuit"] = dict(cx=int(tq.count_ops().get("cx", 0)), depth=int(tq.depth()))
    log("QAOA under noise done")

    # ---------------- 7. end to end: classifier -> detour plan -> delay that really happened ----------------
    df = data["df"]
    test = df[df["date_time"] >= "2017-01-01"]
    Xall = D.frame_to_angles(test)
    V = test["traffic_volume"].to_numpy() / thr
    busy = V > 1
    R["endtoend"] = endtoend(V, busy, {
        "Never divert": np.zeros(len(V)),
        "Always plan for busy": np.ones(len(V)),
        "Re-uploading VQC (ours)": final.predict_proba(Xall),
        "MLP (16-16)": clf["MLP (16-16)"].predict_proba(classical_inputs(Xall))[:, 1],
        "Random forest": clf["Random forest"].predict_proba(classical_inputs(Xall))[:, 1],
        "Correct label (perfect classifier)": busy.astype(float),
    }, VB, VQ)
    R["endtoend_hours"] = int(len(V))
    log("end to end", {r["policy"]: round(r["share_of_possible"], 3) for r in R["endtoend"]})
    # same on the noise subset, with noisy classifier outputs
    Vn = vn / thr
    for row in R["noise"]:
        P_raw, P_mit = noisy_P[row["level"]]
        e = {r["policy"]: r for r in endtoend(Vn, Vn > 1, {"raw": P_raw, "mitigated": P_mit}, VB, VQ)}
        row["saving_raw"], row["saving_mitigated"] = e["raw"]["share_of_possible"], e["mitigated"]["share_of_possible"]
    # QAOA as the solver inside the loop, on busy-predicted test hours (ideal and noisy device)
    idx = rng.choice(np.flatnonzero(final.predict_proba(Xall) > 0.5), 10 if quick else 60, replace=False)
    loop = []
    for k, i in enumerate(idx):
        inst = routing.make_instance(final.predict_proba(Xall[i:i + 1])[0], VB, VQ)
        xe, _, _ = routing.solve_exact(inst)
        q = routing.run_qaoa(inst, p=2, restarts=2, seed=k)
        xq, _ = routing.sample_plan(q, shots=256, seed=k)
        c = noise.run([routing.qaoa_circuit(inst, q["params"][:2], q["params"][2:])], 8, 1, shots=256, seed=k)[0]
        Xs, _ = routing.counts_to_plans(c, 8)
        xn = Xs[np.argmin(routing.qubo_cost(inst, Xs))]
        real = lambda x: float(routing.realised_cost(x, V[i], busy[i])[0])
        loop.append(dict(same_ideal=bool((xq == xe).all()), same_noisy=bool((xn == xe).all()),
                         extra_ideal=real(xq) - real(xe), extra_noisy=real(xn) - real(xe), exact=real(xe)))
    R["qaoa_in_loop"] = loop
    log("QAOA in the loop", np.mean([l["same_ideal"] for l in loop]), np.mean([l["same_noisy"] for l in loop]))

    # scaling of exact simulation
    sc = []
    for n in range(4, 17, 2):
        inst = routing.make_instance(0.9, VB, VQ, shares=rng.dirichlet(np.ones(n)))
        Xn_ = routing._bit_order(n); c = routing.qubo_cost(inst, Xn_); c = (c - c.min()) / (c.max() - c.min())
        t = time.perf_counter(); routing.qaoa_state(c, [0.5, 0.4], [0.6, 0.3], n); tq = time.perf_counter() - t
        t = time.perf_counter(); routing.solve_exact(inst); te = time.perf_counter() - t
        sc.append(dict(n=n, qaoa_eval_ms=tq * 1000, exact_ms=te * 1000))
    R["qaoa_scaling"] = sc


def endtoend(V, busy, policies, VB, VQ):
    """Plan detours for every hour from each policy's P(busy) (exact solver), then score each plan with the
    delay it really causes given the hour's real volume and state. Detours are only released when P > 0.5."""
    X8 = routing.all_bitstrings(8)
    Jb = routing.state_cost(X8, VB, routing.C_BUSY); Jq = routing.state_cost(X8, VQ, routing.C_QUIET)
    cap = np.where(busy, routing.C_BUSY, routing.C_QUIET)
    def real(plan_idx):
        a = V * (X8[plan_idx] @ routing.SHARES)
        return (V - a) ** 2 / cap + a ** 2 / routing.C_ALT + routing.TAU * a
    allJ = (V[:, None] - V[:, None] * (X8 @ routing.SHARES)[None]) ** 2 / cap[:, None] \
        + (V[:, None] * (X8 @ routing.SHARES)[None]) ** 2 / routing.C_ALT + routing.TAU * V[:, None] * (X8 @ routing.SHARES)[None]
    never, oracle = real(np.zeros(len(V), int)).sum(), allJ.min(1).sum()
    rows = []
    for name, P in policies.items():
        plan = np.argmin(P[:, None] * Jb[None] + (1 - P[:, None]) * Jq[None], axis=1)
        plan[P <= 0.5] = 0                                       # classical gate: no detours unless "busy"
        tot = real(plan).sum()
        rows.append(dict(policy=name, delay=float(tot), saved=float((never - tot) / never),
                         share_of_possible=float((never - tot) / (never - oracle)),
                         hours_with_detours=int((plan > 0).sum())))
    rows.append(dict(policy="Best possible (knows the real traffic)", delay=float(oracle), saved=float((never - oracle) / never),
                     share_of_possible=1.0, hours_with_detours=int((allJ.argmin(1) > 0).sum())))
    return rows


# ======================================================================
def make_figures(R, final):
    import pandas as pd
    comp = pd.DataFrame(R["comparison"])

    # learning curve
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    styles = {"Re-uploading VQC (ours)": ("#7c3aed", "o-", 2.4), "Standard VQC": ("#c4b5fd", "o--", 1.4),
              "Quantum-kernel SVM": ("#a78bfa", "s:", 1.4), "Logistic regression": ("#94a3b8", "^-", 1.2),
              "SVM (RBF)": ("#38bdf8", "v-", 1.2), "Random forest": ("#0284c7", "D-", 1.2),
              "MLP (16-16)": ("#0f172a", "x-", 1.2)}
    for name, (c, st, lw) in styles.items():
        g = comp[comp.model == name].groupby("n")["acc"]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std().fillna(0), fmt=st, color=c, lw=lw, ms=4, capsize=2,
                    label=name)
    ax.set_xscale("log"); ax.set_xlabel("training snapshots"); ax.set_ylabel("test accuracy (2017–18)")
    ax.legend(fontsize=7, ncol=2, frameon=False); ax.grid(alpha=0.25)
    fig.savefig(FIG / "learning_curve.png"); plt.close(fig)

    # ablation
    ab = pd.DataFrame(R["ablation"]).groupby("variant", sort=False)["acc"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cols = ["#7c3aed" if "ours" in v else "#a78bfa" if v.startswith("  ") else "#94a3b8" for v in ab.index]
    ax.barh(range(len(ab))[::-1], ab["mean"], xerr=ab["std"], color=cols, capsize=2)
    ax.set_yticks(range(len(ab))[::-1]); ax.set_yticklabels([v.strip() for v in ab.index], fontsize=8)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("test accuracy (n=500, 3 seeds)")
    fig.savefig(FIG / "ablation.png"); plt.close(fig)

    # noise: classifier accuracy, probability error, QAOA quality vs device noise level
    nz = pd.DataFrame(R["noise"]); qn = pd.DataFrame(R["qaoa_noise"]).groupby("level").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    axes[0].plot(nz.level, nz.acc_raw, "o-", color="#94a3b8", label="raw")
    axes[0].plot(nz.level, nz.acc_mitigated, "s-", color="#7c3aed", label="mitigated")
    axes[0].axhline(R["noise_ideal_acc"], ls="--", color="k", lw=1, label="noise-free")
    axes[0].set_ylim(0.85, 0.97); axes[0].set_xlabel("device noise level (1 = today)"); axes[0].set_ylabel("classifier accuracy")
    axes[0].legend(fontsize=7)
    axes[1].plot(nz.level, nz.prob_err_raw, "o-", color="#94a3b8", label="raw")
    axes[1].plot(nz.level, nz.prob_err_readout, "^-", color="#38bdf8", label="readout fixed")
    axes[1].plot(nz.level, nz.prob_err_mitigated, "s-", color="#7c3aed", label="readout + rescaling")
    axes[1].set_xlabel("device noise level"); axes[1].set_ylabel("mean |P_noisy − P_ideal|"); axes[1].legend(fontsize=7)
    rnd = pd.DataFrame(R["qaoa"]); rnd = rnd[rnd.method == "Random guess (p=0)"]["approx_ratio"].mean()
    axes[2].plot(qn.index, qn.approx_ratio, "D-", color="#7c3aed", label="QAOA p=2")
    axes[2].axhline(rnd, ls=":", color="#94a3b8", label="random plan")
    axes[2].set_xlabel("device noise level"); axes[2].set_ylabel("QAOA approximation ratio"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "noise.png"); plt.close(fig)

    # end to end: share of the possible delay reduction achieved by each policy
    ee = pd.DataFrame(R["endtoend"])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    col = ["#94a3b8" if "Never" in p or "Always" in p else "#7c3aed" if "VQC" in p else "#0f766e" if "Best" in p or "Correct" in p
           else "#0ea5e9" for p in ee.policy]
    ax.barh(ee.policy[::-1], 100 * ee.share_of_possible[::-1], color=col[::-1])
    for i, v in enumerate(100 * ee.share_of_possible[::-1]):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 112); ax.set_xlabel("share of the possible delay reduction achieved (%)")
    fig.savefig(FIG / "endtoend.png"); plt.close(fig)

    # latency
    lat = R["latency"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names = list(lat); med = [lat[n]["median_ms"] for n in names]
    ax.barh(names[::-1], med[::-1], color=["#0ea5e9" if "classical" in n or n in ("Random forest", "MLP")
                                          else "#7c3aed" for n in names[::-1]])
    ax.axvline(1000, color="red", ls="--", lw=1); ax.text(1000, 0, " 1 s budget", color="red", fontsize=8, va="bottom")
    ax.set_xscale("log"); ax.set_xlabel("median latency per snapshot (ms, log scale)")
    fig.savefig(FIG / "latency.png"); plt.close(fig)

    # QAOA
    qa = pd.DataFrame(R["qaoa"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    order = ["Random guess (p=0)"] + [m for m in qa.method.unique() if m.startswith("QAOA")]
    g = qa[qa.method.isin(order)].groupby("method", sort=False)
    axes[0].bar(order, [g.get_group(m)["approx_ratio"].mean() for m in order], color="#7c3aed",
                yerr=[g.get_group(m)["approx_ratio"].std() for m in order], capsize=2)
    axes[0].set_ylabel("approximation ratio ⟨r⟩"); axes[0].set_ylim(0.5, 1.0)
    axes[0].set_xticks(range(len(order))); axes[0].set_xticklabels([o.replace("Random guess (p=0)", "p=0").replace("QAOA ", "") for o in order])
    axes[1].bar(order, [g.get_group(m)["p_near"].mean() for m in order], color="#a78bfa")
    axes[1].set_ylabel("P(plan within 2% of optimum)")
    axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels([o.replace("Random guess (p=0)", "p=0").replace("QAOA ", "") for o in order])
    fig.tight_layout(); fig.savefig(FIG / "qaoa.png"); plt.close(fig)

    # circuit (one layer, symbolic)
    from quantum_its.circuits import CircuitSpec as CS
    qc = to_qiskit(CS(layers=1, scaling=True), None, None, symbolic=True)
    qc.draw("mpl", style="clifford").savefig(FIG / "circuit.png"); plt.close("all")

    # training loss
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.plot(R["final"]["loss"], color="#7c3aed"); ax.set_xlabel("Adam step"); ax.set_ylabel("BCE loss")
    fig.savefig(FIG / "loss.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()
    main(a.quick, a.deploy)
