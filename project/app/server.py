"""QuantumITS web app: FastAPI backend serving the hybrid models and the dashboard.

    uvicorn app.server:app --reload        (from the project/ folder)
"""
import json
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantum_its import data as D, noise, routing  # noqa: E402
from quantum_its.models import VQC, classical_inputs  # noqa: E402

app = FastAPI(title="QuantumITS")

# ---------- load once at start-up ----------
SPEC = json.loads((ROOT / "models" / "vqc_spec.json").read_text())
VQC_MODEL = VQC.from_state({**dict(np.load(ROOT / "models" / "vqc.npz")), "spec": SPEC})
with open(ROOT / "models" / "classical.pkl", "rb") as f:
    _m = pickle.load(f)
CLASSICAL, QKSVM = _m["classical"], _m["qksvm"]
RESULTS = json.loads((ROOT / "results" / "results.json").read_text())
DF = D.clean(D.load_raw())
THRESHOLD = RESULTS["dataset"]["threshold"]
VB, VQ = routing.typical_volumes(DF[DF["date_time"] < "2017-01-01"]["traffic_volume"], THRESHOLD)
SHRINK = {r["level"]: r["shrink"] for r in RESULTS["noise"]}           # calibrated rescaling factor per noise level

# ---------- ground truth: every hour of the 2017-18 test period, with each model's answer ----------
# None of these hours were used for training (the models learned from 2012-16 only).
MODEL_COLS = {"vqc": "Re-uploading VQC", "qk": "Quantum-kernel SVM", "mlp": "MLP (16-16)",
              "rf": "Random forest", "svm": "SVM (RBF)", "lr": "Logistic regression"}
TEST = DF[DF["date_time"] >= "2017-01-01"].copy()
_X = D.frame_to_angles(TEST)
TEST["actual"] = (TEST["traffic_volume"] > THRESHOLD).astype(int)
TEST["p_vqc"] = VQC_MODEL.predict_proba(_X)
TEST["p_qk"] = QKSVM.predict_proba(_X)
for _k, _name in [("mlp", "MLP (16-16)"), ("rf", "Random forest"), ("svm", "SVM (RBF)"), ("lr", "Logistic regression")]:
    TEST[f"p_{_k}"] = CLASSICAL[_name].predict_proba(classical_inputs(_X))[:, 1]
TEST["temp_c"] = TEST["temp"] - 273.15
TEST["date"] = TEST["date_time"].dt.strftime("%Y-%m-%d")

# detour plan chosen from the quantum classifier's P(busy), and the delay it really caused that hour
_plans = routing.best_plans(TEST["p_vqc"].to_numpy(), VB, VQ)
_real = routing.realised_all(TEST["traffic_volume"].to_numpy() / THRESHOLD, TEST["actual"].to_numpy() == 1)
_X8 = routing.all_bitstrings(8)
TEST["ramps"] = ["".join(str(int(b)) for b in x) for x in _plans]
TEST["delay_never"] = _real[:, 0]
TEST["delay_ours"] = _real[np.arange(len(TEST)), (_plans * (2 ** np.arange(8))).sum(1).astype(int)]
TEST["delay_best"] = _real.min(1)
TEST["ramps_best"] = ["".join(str(int(b)) for b in _X8[i]) for i in _real.argmin(1)]


def _score(k):
    pred = TEST[f"p_{k}"] > 0.5
    act = TEST["actual"] == 1
    return dict(key=k, model=MODEL_COLS[k], family="quantum" if k in ("vqc", "qk") else "classical",
                acc=float((pred == act).mean()), tp=int((pred & act).sum()), tn=int((~pred & ~act).sum()),
                fp=int((pred & ~act).sum()), fn=int((~pred & act).sum()))


SCORES = [_score(k) for k in MODEL_COLS]
DATES = set(TEST["date"])
_wrong = TEST[(TEST["p_vqc"] > 0.5) != (TEST["actual"] == 1)].groupby("date").size()
MISTAKE_DATES = _wrong[_wrong >= 2].index.to_numpy()


def _row(r):
    return dict(time=r.date_time.strftime("%H:%M"), date=r.date, weekday=r.date_time.strftime("%a"),
                hour=int(r.hour), day_type=int(r.day_type), weather=r.weather_main, temp_c=round(float(r.temp_c), 1),
                cars=int(r.traffic_volume), actual=int(r.actual),
                p={k: float(getattr(r, f"p_{k}")) for k in MODEL_COLS},
                ramps=r.ramps, ramps_best=r.ramps_best,
                delay=dict(never=float(r.delay_never), ours=float(r.delay_ours), best=float(r.delay_best)))


def _savings(rows):
    never, ours, best = (float(rows[c].sum()) for c in ("delay_never", "delay_ours", "delay_best"))
    return dict(never=never, ours=ours, best=best, saved=(never - ours) / never if never else 0.0,
                best_saved=(never - best) / never if never else 0.0)


class Scenario(BaseModel):
    hour: int = Field(17, ge=0, le=23)
    day_type: int = Field(0, ge=0, le=1)
    temp_c: float = Field(15.0, ge=-30, le=38)
    weather: int = Field(1, ge=0, le=4)
    emergency: int = Field(-1, ge=-1, le=7)          # ramp with an ambulance, -1 = none
    p: int = Field(2, ge=1, le=3)                    # QAOA depth
    noise_level: float = Field(0, ge=0, le=4)        # 0 = ideal simulator, 1 = today's device, 2 = twice as noisy


def _timed(fn):
    t = time.perf_counter(); out = fn(); return out, (time.perf_counter() - t) * 1000


@lru_cache(maxsize=8)
def _readout(level):
    return noise.readout_matrix(level)


def _bits(x):
    return "".join(str(int(b)) for b in x)


@app.post("/api/plan")
def plan(s: Scenario):
    """One input -> congestion prediction (all models) -> detour plan built from the quantum model's P(busy)."""
    t0 = time.perf_counter()
    x, t_pre = _timed(lambda: D.to_angles(s.hour, s.day_type, s.temp_c + 273.15, s.weather)[None, :])
    preds = []
    p, t = _timed(lambda: float(VQC_MODEL.predict_proba(x)[0]))
    preds.append(dict(model="Re-uploading VQC", family="quantum", p=p, ms=t))
    p, t = _timed(lambda: float(QKSVM.predict_proba(x)[0]))
    preds.append(dict(model="Quantum-kernel SVM", family="quantum", p=p, ms=t))
    for name in ("Random forest", "MLP (16-16)", "Logistic regression"):
        p, t = _timed(lambda: float(CLASSICAL[name].predict_proba(classical_inputs(x))[0, 1]))
        preds.append(dict(model=name, family="classical", p=p, ms=t))

    # the quantum classifier on the chosen hardware
    hw = None
    P = preds[0]["p"]
    if s.noise_level > 0:
        lv = float(s.noise_level)
        M = _readout(lv)
        (ev_raw, ev_ro), t = _timed(lambda: (noise.noisy_expval(VQC_MODEL, x, lv)[0],
                                             noise.noisy_expval(VQC_MODEL, x, lv, M=M)[0]))
        f = SHRINK.get(lv) or SHRINK.get(int(lv)) or 1.0
        p_raw, p_mit = float(noise.decide(VQC_MODEL, ev_raw)), float(noise.decide(VQC_MODEL, ev_ro, f))
        hw = dict(level=lv, p_raw=p_raw, p_mitigated=p_mit, shrink=f, ms=t)
        P = p_mit                                               # the plan uses the mitigated noisy prediction

    # detour planning (classical gate first: detours only when the prediction says "busy")
    busy = P > 0.5
    emergency = None if s.emergency < 0 else s.emergency
    volume = P * VB + (1 - P) * VQ                              # expected corridor volume, threshold units
    route = dict(released=busy, p_busy=P, expected_cars=volume * THRESHOLD,
                 ramp_cars=(volume * THRESHOLD * routing.SHARES).tolist(), shares=routing.SHARES.tolist())
    if busy:
        inst = routing.make_instance(P, VB, VQ, emergency=emergency)
        q, tq = _timed(lambda: routing.run_qaoa(inst, p=s.p, restarts=1, seed=0))
        if s.noise_level > 0:                                   # sample the trained circuit on the noisy device
            c, tn = _timed(lambda: noise.run([routing.qaoa_circuit(inst, q["params"][:s.p], q["params"][s.p:])],
                                             8, float(s.noise_level), shots=256, seed=0)[0])
            Xs, w = routing.counts_to_plans(c, 8)
            cs = routing.qubo_cost(inst, Xs)
            xq, cq = Xs[np.argmin(cs)], float(cs.min()); tq += tn
            top_X, top_p = Xs, w
        else:
            xq, cq = routing.sample_plan(q, shots=256, seed=0)
            top_X, top_p = q["X"], q["probs"]
        (xe, ce, _), te = _timed(lambda: routing.solve_exact(inst))
        (xg, cg, _), tg = _timed(lambda: routing.solve_greedy(inst))
        costs = routing.qubo_cost(inst, routing.all_bitstrings(8))
        order = np.argsort(top_p)[::-1][:12]
        top = [dict(bits=_bits(top_X[i]), prob=float(top_p[i]), best=bool(np.isclose(routing.qubo_cost(inst, top_X[i])[0], ce)))
               for i in order]
        if not any(t["best"] for t in top):                     # always show the best plan in the chart
            ib = next((i for i in range(len(top_X)) if np.isclose(routing.qubo_cost(inst, top_X[i])[0], ce)), None)
            top = top[:11] + [dict(bits=_bits(xe), prob=float(top_p[ib]) if ib is not None else 0.0, best=True)]
        route.update(
            plans={"QAOA": dict(x=xq.tolist(), cost=float(cq), ms=tq), "Exact": dict(x=xe.tolist(), cost=float(ce), ms=te),
                   "Greedy": dict(x=xg.tolist(), cost=float(cg), ms=tg)},
            no_detour_cost=float(costs[0]), rank=int((costs < cq - 1e-12).sum()) + 1, n_plans=len(costs),
            p_opt=float(sum(t["prob"] for t in top if t["best"])), top=top, emergency=emergency)
        x_plan = xq
    else:
        x_plan = np.zeros(8)

    # ground truth: real 2017-18 hours with the same conditions, and the delay our plan would really cause there
    sim = TEST[(TEST["hour"] == s.hour) & (TEST["day_type"] == s.day_type) & (TEST["weather"] == s.weather)
               & ((TEST["temp_c"] - s.temp_c).abs() <= 5)]
    similar = dict(n=len(sim))
    if len(sim):
        V = sim["traffic_volume"].to_numpy() / THRESHOLD
        R = routing.realised_all(V, sim["actual"].to_numpy() == 1)
        idx = int((x_plan * (2 ** np.arange(8))).sum())
        never, ours, best = R[:, 0].sum(), R[:, idx].sum(), R.min(1).sum()
        similar.update(busy=int(sim["actual"].sum()), avg_cars=float(sim["traffic_volume"].mean()),
                       vqc_correct=int(((sim["p_vqc"] > 0.5) == (sim["actual"] == 1)).sum()),
                       saved=float((never - ours) / never), best_saved=float((never - best) / never),
                       examples=[_row(r) for r in sim.sample(min(6, len(sim)), random_state=0).sort_values("date_time").itertuples()])

    return dict(angles=x[0].tolist(), predictions=preds, hardware=hw, decision="Busy" if busy else "Not busy",
                threshold=THRESHOLD, route=route, similar=similar,
                state_probs=VQC_MODEL.state_probabilities(x[0]).tolist(), expval_z0=float(VQC_MODEL.expval(x)[0]),
                preprocess_ms=t_pre, total_ms=(time.perf_counter() - t0) * 1000)


@app.get("/api/overview")
def overview():
    by_hour = DF.groupby(["day_type", "hour"])["traffic_volume"].mean().unstack(0)
    sample = DF.sample(6, random_state=3).sort_values("date_time")
    return dict(
        threshold=THRESHOLD, n_rows=len(DF), n_train=int((DF["date_time"] < "2017-01-01").sum()), n_test=len(TEST),
        first=DF["date_time"].min().strftime("%b %Y"), last=DF["date_time"].max().strftime("%b %Y"),
        weekday=by_hour[0].round().tolist(), weekend=by_hour[1].round().tolist(),
        sample=[dict(when=r.date_time.strftime("%a %d %b %Y, %H:%M"), weather=r.weather_main,
                     temp_c=round(r.temp - 273.15, 1), holiday=bool(r.day_type), cars=int(r.traffic_volume),
                     busy=bool(r.traffic_volume > THRESHOLD)) for r in sample.itertuples()],
        scores=SCORES, test_first=TEST["date"].min(), test_last=TEST["date"].max(),
        v_busy=VB * THRESHOLD, v_quiet=VQ * THRESHOLD)


@app.get("/api/day")
def day(date: str | None = None, seed: int | None = None, mistakes: bool = False):
    """All test-period hours of one real day: real traffic, every model's prediction, and the detour plan.
    No date (or an unknown one) picks a random day; mistakes=true picks a day where the VQC was wrong at least twice."""
    if date is None or date not in DATES:
        dates = MISTAKE_DATES if mistakes else TEST["date"].unique()
        date = dates[np.random.default_rng(seed).integers(len(dates))]
    rows = TEST[TEST["date"] == date].sort_values("date_time")
    return dict(date=date, label=rows["date_time"].iloc[0].strftime("%A %d %B %Y"),
                holiday=bool(rows["day_type"].iloc[0]), threshold=THRESHOLD, savings=_savings(rows),
                rows=[_row(r) for r in rows.itertuples()])


@app.get("/api/results")
def results():
    return RESULTS


app.mount("/figures", StaticFiles(directory=ROOT / "figures"), name="figures")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
