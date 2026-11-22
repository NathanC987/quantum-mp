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
    probs = VQC_MODEL.state_probabilities(x[0])
    ev = float(VQC_MODEL.expval(x)[0])
    # what the platform would do with the decision (classical stage)
    congested = out[0]["p"] > 0.5
    action = ("Extend green on the I-94 approach, publish alternate-route advice"
              if congested else "Keep the normal signal plan")
    return dict(angles=x[0].tolist(), predictions=out, state_probs=probs.tolist(), expval_z0=ev,
                decision="Congested" if congested else "Free-flowing", action=action,
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
    return dict(
        demands=demands.tolist(), emergency=r.emergency,
        plans={"QAOA": dict(x=xq.tolist(), cost=float(cq), ms=tq, alt_share=float(xq @ share)),
               "Exact": dict(x=xe.tolist(), cost=float(ce), ms=te, alt_share=float(xe @ share)),
               "Greedy": dict(x=xg.tolist(), cost=float(cg), ms=tg, alt_share=float(xg @ share))},
        approx_ratio=q["approx_ratio"], p_opt=q["p_opt"],
        top_states=[dict(bits="".join(str(int(b)) for b in q["X"][i]), prob=float(q["probs"][i]),
                         cost=float(q["costs"][i])) for i in top])


@app.get("/api/qec")
def qec_rate(p: float = 0.02):
    p = float(np.clip(p, 0.0005, 0.3))
    return {c: dict(p_logical=qec.logical_error_rate(c, round(p, 4), shots=6000),
                    qubits=qec.physical_qubits(c)) for c in qec.CODES}


@app.get("/api/results")
def results():
    return RESULTS


app.mount("/figures", StaticFiles(directory=ROOT / "figures"), name="figures")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
