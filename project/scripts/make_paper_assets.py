"""Writes ../paper/numbers.tex (LaTeX macros with every reported number) and copies figures to ../paper/."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper"; PAPER.mkdir(exist_ok=True)
R = json.loads((ROOT / "results" / "results.json").read_text())

# ---------- pipeline figure ----------
fig, ax = plt.subplots(figsize=(7.2, 1.55)); ax.axis("off"); ax.set_xlim(0, 8); ax.set_ylim(0, 1.6)
stages = [("Sensors &\nweather", "c"), ("Clean +\nfeatures", "c"), ("Encode\n$R_Y(w x)$", "q"), ("Ansatz\n$U(\\theta)$ ×L", "q"),
          ("Measure\n$\\langle Z_0\\rangle$", "q"), ("Decide\n$\\sigma(a\\langle Z\\rangle+b)$", "c"), ("QAOA\nroutes", "q"),
          ("Check +\nactuate", "c")]
col = {"q": ("#ede9fe", "#7c3aed"), "c": ("#e0f2fe", "#0284c7")}
for i, (t, k) in enumerate(stages):
    ax.add_patch(FancyBboxPatch((i + 0.06, 0.35), 0.88, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=col[k][0], ec=col[k][1], lw=1.2))
    ax.text(i + 0.5, 0.82, t, ha="center", va="center", fontsize=7.3)
    if i < 7:
        ax.annotate("", (i + 1.06, 0.82), (i + 0.94, 0.82), arrowprops=dict(arrowstyle="->", lw=0.9))
ax.add_patch(FancyBboxPatch((2.0, 0.05), 3.0, 0.18, boxstyle="round,pad=0.01", fc="#7c3aed", ec="none"))
ax.text(3.5, 0.14, "hybrid loop: Adam updates θ, w, a, b", color="white", ha="center", va="center", fontsize=6.5)
fig.savefig(PAPER / "pipeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)

for f in ["learning_curve", "ablation", "noise", "qaoa", "qec", "circuit"]:
    shutil.copy(ROOT / "figures" / f"{f}.png", PAPER / f"{f}.png")

# ---------- numbers ----------
comp = pd.DataFrame(R["comparison"])
g = comp.groupby(["model", "n"])
acc, acc_sd, auc, f1, rec = (g.acc.mean(), g.acc.std(), g.auc.mean(), g.f1.mean(), g.recall.mean())
fit_s = g.fit_s.mean()
abl = pd.DataFrame(R["ablation"]).groupby("variant", sort=False)
qa = pd.DataFrame(R["qaoa"]).groupby("method", sort=False).mean(numeric_only=True)
qe = pd.DataFrame(R["qec"]).set_index("p")
nz = pd.DataFrame(R["noise"]).set_index("p2")
lat = R["latency"]; d = R["dataset"]

M = {}
def put(k, v): M[k] = v

put("Nrows", f"{d['n_rows']:,}".replace(",", "{,}")); put("Npool", f"{d['n_train_pool']:,}".replace(",", "{,}"))
put("Ntest", f"{d['n_test_period']:,}".replace(",", "{,}")); put("Nused", f"{d['n_test_used']:,}".replace(",", "{,}"))
put("Thr", f"{d['threshold']:.0f}")
short = {"Re-uploading VQC (ours)": "Ours", "Standard VQC": "Std", "Quantum-kernel SVM": "QK",
         "Logistic regression": "LR", "SVM (RBF)": "SVM", "Random forest": "RF", "MLP (16-16)": "MLP"}
for m, s in short.items():
    for n, w in [(100, "C"), (500, "D"), (1000, "M"), (2000, "MM")]:
        put(f"acc{s}{w}", f"{acc[(m, n)]:.3f}"); put(f"sd{s}{w}", f"{acc_sd[(m, n)]:.3f}")
    put(f"auc{s}", f"{auc[(m, 1000)]:.3f}"); put(f"fone{s}", f"{f1[(m, 1000)]:.3f}"); put(f"rec{s}", f"{rec[(m, 1000)]:.3f}")
    put(f"fit{s}", f"{fit_s[(m, 1000)]:.1f}")
put("accOursFifty", f"{acc[('Re-uploading VQC (ours)', 50)]:.3f}")
a = abl.acc.mean(); asd = abl.acc.std(); pn = abl.n_params.first()
for i, v in enumerate(a.index):
    k = "abl" + "ABCDEFGHIJKLMN"[i]
    put(k, f"{a[v]:.3f}"); put(k + "sd", f"{asd[v]:.3f}"); put(k + "p", f"{pn[v]}")
put("noiseIdeal", f"{R['noise_ideal_acc']:.3f}")
for p2, tag in [(0.005, "A"), (0.01, "B"), (0.02, "C"), (0.05, "D")]:
    put(f"noiseRaw{tag}", f"{nz.loc[p2, 'acc_raw']:.3f}"); put(f"noiseMit{tag}", f"{nz.loc[p2, 'acc_mitigated']:.3f}")
    put(f"shrink{tag}", f"{nz.loc[p2, 'shrink']:.2f}")
    put(f"probErrRaw{tag}", f"{nz.loc[p2, 'prob_err_raw']:.3f}"); put(f"probErrMit{tag}", f"{nz.loc[p2, 'prob_err_mitigated']:.3f}")
sh = pd.DataFrame(R["shots"]).set_index("shots")
put("shotsSixteen", f"{sh.loc[16, 'acc']:.3f}"); put("shotsK", f"{sh.loc[1024, 'acc']:.3f}")
names = {"Preprocessing (classical)": "Pre", "VQC – exact simulator": "Vqc", "VQC – Qiskit Aer, 1024 shots": "Aer",
         "Quantum-kernel SVM": "Qk", "Random forest": "Rf", "MLP": "Mlp"}
for k, s in names.items():
    put(f"lat{s}", f"{lat[k]['median_ms']:.2f}"); put(f"latp{s}", f"{lat[k]['p95_ms']:.2f}")
put("depth", R["circuit"]["depth"]); put("ncx", R["circuit"]["cx"]); put("nparams", R["circuit"]["n_params"])
put("simdiff", f"{R['check_sim_vs_qiskit_max_abs_diff']:.0e}".replace("e-", "\\times10^{-") + "}")
put("qubodiff", f"{R['qubo_check_max_abs_diff']:.0e}".replace("e-", "\\times10^{-") + "}")
for m, s in [("Random guess (p=0)", "R"), ("QAOA p=1", "One"), ("QAOA p=2", "Two"), ("QAOA p=3", "Three"), ("QAOA p=4", "Four")]:
    put(f"ar{s}", f"{qa.loc[m, 'approx_ratio']:.3f}"); put(f"popt{s}", f"{100 * qa.loc[m, 'p_opt']:.2f}")
    put(f"pnear{s}", f"{100 * qa.loc[m, 'p_near']:.1f}")
    if s != "R":
        put(f"gap{s}", f"{100 * qa.loc[m, 'gap']:.2f}"); put(f"ms{s}", f"{qa.loc[m, 'ms']:.0f}")
for m, s in [("Exact enumeration", "Ex"), ("Greedy", "Gr"), ("Simulated annealing", "Sa")]:
    put(f"gap{s}", f"{100 * qa.loc[m, 'gap']:.2f}"); put(f"ms{s}", f"{qa.loc[m, 'ms']:.2f}")
for p, tag in [(0.001, "A"), (0.01, "B"), (0.05, "C"), (0.12, "D")]:
    for c, s in [("unprotected", "U"), ("bit_flip", "B"), ("phase_flip", "P"), ("shor", "S")]:
        put(f"qec{s}{tag}", f"{qe.loc[p, c]:.4f}")
put("shorGain", f"{qe.loc[0.01, 'unprotected'] / max(qe.loc[0.01, 'shor'], 1e-6):.0f}")

(PAPER / "numbers.tex").write_text("% auto-generated by project/scripts/make_paper_assets.py\n" +
                                   "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
print(f"{len(M)} macros written; ablation order:", list(a.index))
