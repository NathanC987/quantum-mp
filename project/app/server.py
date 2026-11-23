"""QuantumITS web app: FastAPI backend serving the hybrid models and the dashboard.

    uvicorn app.server:app --reload        (from the project/ folder)
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantum_its import data as D, qec, routing  # noqa: E402
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
                p={k: float(getattr(r, f"p_{k}")) for k in MODEL_COLS})


class Snapshot(BaseModel):
    hour: int = Field(17, ge=0, le=23)
    day_type: int = Field(0, ge=0, le=1)
    temp_c: float = Field(15.0, ge=-30, le=38)
    weather: int = Field(1, ge=0, le=4)


class RouteRequest(BaseModel):
    hour: int = Field(17, ge=0, le=23)
    emergency: int = Field(0, ge=0, le=7)
    p: int = Field(1, ge=1, le=3)
    seed: int = 0


def _timed(fn):
    t = time.perf_counter(); out = fn(); return out, (time.perf_counter() - t) * 1000


@app.post("/api/predict")
def predict(s: Snapshot):
    t0 = time.perf_counter()
    x, t_pre = _timed(lambda: D.to_angles(s.hour, s.day_type, s.temp_c + 273.15, s.weather)[None, :])
    out = []
    p, t = _timed(lambda: float(VQC_MODEL.predict_proba(x)[0]))
    out.append(dict(model="Re-uploading VQC", family="quantum", p=p, ms=t))
    p, t = _timed(lambda: float(QKSVM.predict_proba(x)[0]))
    out.append(dict(model="Quantum-kernel SVM", family="quantum", p=p, ms=t))
    for name in ("Random forest", "MLP (16-16)", "Logistic regression"):
        p, t = _timed(lambda: float(CLASSICAL[name].predict_proba(classical_inputs(x))[0, 1]))
        out.append(dict(model=name, family="classical", p=p, ms=t))
    # ground truth for a what-if: real 2017-18 hours with the same conditions
    sim = TEST[(TEST["hour"] == s.hour) & (TEST["day_type"] == s.day_type) & (TEST["weather"] == s.weather)
               & ((TEST["temp_c"] - s.temp_c).abs() <= 5)]
    similar = dict(n=len(sim), busy=int(sim["actual"].sum()), avg_cars=float(sim["traffic_volume"].mean()) if len(sim) else None,
                   vqc_correct=int(((sim["p_vqc"] > 0.5) == (sim["actual"] == 1)).sum()),
                   examples=[_row(r) for r in sim.sample(min(6, len(sim)), random_state=0).sort_values("date_time").itertuples()])
    probs = VQC_MODEL.state_probabilities(x[0])
    ev = float(VQC_MODEL.expval(x)[0])
    # what the platform would do with the decision (classical stage)
    congested = out[0]["p"] > 0.5
    action = ("Give the highway approach longer green lights and suggest detours to drivers"
              if congested else "Keep the normal traffic-light timings")
    return dict(angles=x[0].tolist(), predictions=out, similar=similar, threshold=THRESHOLD, state_probs=probs.tolist(), expval_z0=ev,
                decision="Busy" if congested else "Not busy", action=action,
                preprocess_ms=t_pre, total_ms=(time.perf_counter() - t0) * 1000)


@app.post("/api/route")
def route(r: RouteRequest):
    rng = np.random.default_rng(r.seed)
    pool = DF[DF["hour"] == r.hour]["traffic_volume"].to_numpy()
    demands = rng.choice(pool, 8, replace=False)
    inst = routing.make_instance(demands, emergency=r.emergency)
    q, tq = _timed(lambda: routing.run_qaoa(inst, p=r.p, restarts=1, seed=r.seed))
    xq, cq = routing.sample_plan(q, shots=256, seed=r.seed)
    (xe, ce, _), te = _timed(lambda: routing.solve_exact(inst))
    (xg, cg, _), tg = _timed(lambda: routing.solve_greedy(inst))
    share = demands / demands.sum()
    top = np.argsort(q["probs"])[::-1][:12]
    best = int(np.argmin(q["costs"]))
    if best not in top:                                         # always show the best plan in the chart
        top = np.append(top[:11], best)
    # rank of the QAOA plan among all 2^8 = 256 possible plans (1 = the best plan)
    rank = int((q["costs"] < cq - 1e-12).sum()) + 1
    return dict(
        demands=demands.tolist(), emergency=r.emergency,
        plans={"QAOA": dict(x=xq.tolist(), cost=float(cq), ms=tq, alt_share=float(xq @ share)),
               "Exact": dict(x=xe.tolist(), cost=float(ce), ms=te, alt_share=float(xe @ share)),
               "Greedy": dict(x=xg.tolist(), cost=float(cg), ms=tg, alt_share=float(xg @ share))},
        approx_ratio=q["approx_ratio"], p_opt=q["p_opt"], rank=rank, n_plans=len(q["costs"]),
        top_states=[dict(bits="".join(str(int(b)) for b in q["X"][i]), prob=float(q["probs"][i]),
                         cost=float(q["costs"][i])) for i in top])


@app.get("/api/qec")
def qec_rate(p: float = 0.02):
    p = float(np.clip(p, 0.0005, 0.3))
    return {c: dict(p_logical=qec.logical_error_rate(c, round(p, 4), shots=6000),
                    qubits=qec.physical_qubits(c)) for c in qec.CODES}


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
        scores=SCORES, test_first=TEST["date"].min(), test_last=TEST["date"].max())


@app.get("/api/day")
def day(date: str | None = None, seed: int | None = None, mistakes: bool = False):
    """All test-period hours of one real day, with the actual traffic and every model's prediction.
    No date (or an unknown one) picks a random day; mistakes=true picks a day where the VQC was wrong at least twice."""
    if date is None or date not in DATES:
        dates = MISTAKE_DATES if mistakes else TEST["date"].unique()
        date = dates[np.random.default_rng(seed).integers(len(dates))]
    rows = TEST[TEST["date"] == date].sort_values("date_time")
    return dict(date=date, label=rows["date_time"].iloc[0].strftime("%A %d %B %Y"),
                holiday=bool(rows["day_type"].iloc[0]), threshold=THRESHOLD, rows=[_row(r) for r in rows.itertuples()])


@app.get("/api/results")
def results():
    return RESULTS


app.mount("/figures", StaticFiles(directory=ROOT / "figures"), name="figures")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
