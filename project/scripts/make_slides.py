"""Builds slides/QuantumITS.pptx (10 slides) from results/results.json and figures/."""
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "results.json").read_text())
FIG = ROOT / "figures"
OUT = ROOT.parent / "slides"; OUT.mkdir(exist_ok=True)

INK, MUTED, Q, C, BG, SOFTQ, SOFTC = (RGBColor(0x18, 0x18, 0x1B), RGBColor(0x6B, 0x72, 0x80), RGBColor(0x7C, 0x3A, 0xED),
                                      RGBColor(0x02, 0x84, 0xC7), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xED, 0xE9, 0xFE),
                                      RGBColor(0xE0, 0xF2, 0xFE))
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]

comp = pd.DataFrame(R["comparison"]); acc = comp.groupby(["model", "n"]).acc.mean().unstack()
abl = pd.DataFrame(R["ablation"]).groupby("variant", sort=False)[["acc", "auc"]].mean()
qa = pd.DataFrame(R["qaoa"]).groupby("method", sort=False).mean(numeric_only=True)
nz = pd.DataFrame(R["noise"]).set_index("level")
qn = pd.DataFrame(R["qaoa_noise"]).groupby("level").mean(numeric_only=True)
ee = pd.DataFrame(R["endtoend"]).set_index("policy")
lat = R["latency"]


def text(slide, x, y, w, h, s, size=18, color=INK, bold=False, align=None, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(s if isinstance(s, list) else [s]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.size = Pt(size); p.font.color.rgb = color; p.font.bold = bold; p.font.name = font
        p.space_after = Pt(6)
        if align: p.alignment = align
    return tb


def bullets(slide, x, y, w, h, items, size=17):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bold, rest = (it.split("|", 1) + [""])[:2] if "|" in it else ("", it)
        if bold:
            r = p.add_run(); r.text = bold; r.font.bold = True; r.font.size = Pt(size); r.font.color.rgb = Q
        r = p.add_run(); r.text = ("  " if bold else "• ") + rest.strip(); r.font.size = Pt(size); r.font.color.rgb = INK
        p.space_after = Pt(10)
    return tb


def header(title, kicker):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12)); bar.fill.solid()
    bar.fill.fore_color.rgb = Q; bar.line.fill.background()
    text(s, 0.6, 0.35, 12, 0.4, kicker.upper(), 12, Q, True)
    text(s, 0.6, 0.65, 12, 0.9, title, 30, INK, True)
    text(s, 0.6, 7.0, 12, 0.3, "QuantumITS · Hybrid quantum–classical intelligent transportation", 10, MUTED)
    return s


def box(s, x, y, w, h, title, body, fill, size=14, align_left=True):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill; sh.line.fill.background(); sh.adjustments[0] = 0.12
    tf = sh.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = title; p.font.bold = True; p.font.size = Pt(size); p.font.color.rgb = INK
    p2 = tf.add_paragraph(); p2.text = body; p2.font.size = Pt(size - 1); p2.font.color.rgb = MUTED
    if align_left:
        p.alignment = p2.alignment = PP_ALIGN.LEFT
    from pptx.enum.text import MSO_ANCHOR
    tf.vertical_anchor = MSO_ANCHOR.TOP if "\n" in body else MSO_ANCHOR.MIDDLE
    return sh


def image(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def table(s, x, y, w, rows, col_w=None, size=12, hl=None):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.35 * len(rows))).table
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.cell(i, j); cell.text = str(v)
            p = cell.text_frame.paragraphs[0]; p.font.size = Pt(size); p.font.bold = i == 0 or (hl is not None and i == hl)
            p.font.color.rgb = BG if i == 0 else INK
            cell.fill.solid(); cell.fill.fore_color.rgb = Q if i == 0 else (SOFTQ if hl is not None and i == hl else BG)
    if col_w:
        for j, cw in enumerate(col_w): t.columns[j].width = Inches(cw)
    return t


best_c = max(["Logistic regression", "SVM (RBF)", "Random forest", "MLP (16-16)"], key=lambda m: acc.loc[m, 1000])
vqc = acc.loc["Re-uploading VQC (ours)", 1000]

# 1 — title (same light style as the other slides)
s = prs.slides.add_slide(BLANK)
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12)); bar.fill.solid()
bar.fill.fore_color.rgb = Q; bar.line.fill.background()
text(s, 0.9, 1.5, 11.5, 0.5, "CASE STUDY 5 · NATIONAL INTELLIGENT TRANSPORTATION SYSTEM", 14, Q, True)
text(s, 0.9, 2.05, 11.5, 1.9, "QuantumITS: Hybrid Quantum–Classical Congestion Prediction that Drives Detour Planning", 38, INK, True)
text(s, 0.9, 4.15, 11.5, 1.0, ["Data re-uploading VQC → P(busy) → QAOA detour plan · tested on a noisy-device model",
                               "Real I-94 traffic + weather data (UCI, 48k hourly records)"], 18, MUTED)
line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(5.45), Inches(1.2), Inches(0.05)); line.fill.solid()
line.fill.fore_color.rgb = Q; line.line.fill.background()
text(s, 0.9, 5.65, 11.5, 0.5, "Rushikesh Shembade  ·  Nathan Cherian", 18, INK, True)
text(s, 0.9, 6.15, 11.5, 0.5, "Department of Computer Engineering, VES. Institute of Technology, Mumbai  ·  Quantum Machine Learning", 14, MUTED)

# 2 — problem
s = header("The problem: four decisions, one second", "Motivation")
bullets(s, 0.6, 1.7, 6.3, 5, [
    "Platform|signals, cameras, GPS, emergency vehicles, weather stations and autonomous cars in one system.",
    "Tasks|predict traffic flow, optimise signal timing, detect accidents, recommend alternate routes.",
    "Constraint|every prediction must arrive within 1 second.",
    "Our question|which stages actually benefit from a quantum or hybrid approach, and which must stay classical?",
])
box(s, 7.3, 1.8, 5.4, 1.45, "Prediction → quantum ML", "Is this junction about to be congested? (variational quantum classifier)", SOFTQ, size=16)
box(s, 7.3, 3.5, 5.4, 1.45, "Routing → quantum optimisation", "Given P(busy), which on-ramps go on a detour? (QUBO + QAOA)", SOFTQ, size=16)
box(s, 7.3, 5.2, 5.4, 1.45, "Reliability → noise + mitigation", "Does it still work with gate, decay and readout errors?", SOFTQ, size=16)

# 3 — data
d = R["dataset"]
s = header("Data and preprocessing", "Dataset")
bullets(s, 0.6, 1.8, 6.4, 5, [
    f"Source|Metro Interstate Traffic Volume (UCI #492 / Kaggle): hourly I-94 volume with weather, 2012–2018.",
    f"Cleaning|removed sensor glitches (9831 mm/h rain, 0 K temperature) and duplicate hours → {d['n_rows']:,} records.",
    f"Label|congested = volume > {d['threshold']:.0f} veh/h (median of the training years).",
    f"Split|chronological. Train 2012–16 ({d['n_train_pool']:,}), test 2017–18 ({d['n_test_period']:,}). No look-ahead.",
])
table(s, 7.3, 1.8, 5.4, [["Feature x_k", "Angle on qubit k"], ["hour of day", "2π·h/24  (cyclic)"],
                         ["weekend / holiday", "0 or π  (basis-like)"], ["temperature", "[0, π]"],
                         ["weather severity 0–4", "[0, π]"]], [2.7, 2.8], 16)
text(s, 7.3, 4.6, 5.5, 2, "Encoding the hour as 2πh/24 makes it periodic: RY(2π) = −I is only a global phase, "
     "so 23:00 sits next to 00:00 with no extra qubit. The weekend flag is basis-encoded as |0⟩ or |1⟩.", 15, MUTED)

# 4 — architecture
s = header("Hybrid architecture: quantum only where it fits", "Method")
stages = [("Sensors", "c"), ("Preprocess", "c"), ("U_enc(x)", "q"), ("U(θ) ×3", "q"), ("Measure ⟨Z₀⟩", "q"),
          ("P(busy) > 0.5?", "c"), ("QAOA detours", "q"), ("Actuation", "c")]
for i, (n, k) in enumerate(stages):
    box(s, 0.6 + i * 1.53, 1.8, 1.43, 0.9, n, "", SOFTQ if k == "q" else SOFTC, align_left=False)
image(s, FIG / "circuit.png", 0.6, 3.0, w=7.4)
bullets(s, 8.3, 3.0, 4.6, 4, [
    "Encoding|RY(w·x_k): each reading on its own qubit, scaled by a trainable weight w.",
    "Re-uploading|encode → RZ·RY·RZ → ring CNOT, repeated L = 3 times. 50 parameters.",
    "Readout|P(congested) = σ(a⟨Z₀⟩ + b): a two-parameter classical head.",
], 14)

# 5 — symbols
s = header("What the conceptual equations mean here", "Formulation")
table(s, 0.6, 1.7, 12.1, [
    ["Symbol", "Meaning in the ITS case study"],
    ["x = [x₁,x₂,x₃,x₄]", "one junction snapshot: hour, day type, temperature, weather"],
    ["|ψ(x)⟩ = U_enc(x)|0000⟩", "readings loaded as RY rotations: a product state with no correlations yet"],
    ["U_φ(x) = exp(−iφ(x)H)", "feature map: H = Z_kZ_l, φ = (π−x_k)(π−x_l) couples pairs such as rush-hour × snow"],
    ["|φ(x)⟩ = U_φ(x)|ψ(x)⟩", "entangled, data-dependent representation. Tested as an ablation"],
    ["U(θ)", "trainable rotations + entanglement. θ learned with Adam on cross-entropy loss"],
    ["measurement → decision", "P = σ(a⟨Z₀⟩+b); if P > 0.5: re-time signals and plan detours with P"],
], [3.8, 8.3], 18)

# 6 — prediction results
s = header("Prediction: re-uploading VQC matches classical models", "Results · congestion")
image(s, FIG / "learning_curve.png", 0.5, 1.6, w=6.6)
ab = abl["acc"]
table(s, 7.4, 1.7, 5.4, [["Model (n = 1000)", "Accuracy"]] + [
    [m, f"{acc.loc[m, 1000]:.3f}"] for m in ["Re-uploading VQC (ours)", "Standard VQC", "Quantum-kernel SVM",
                                             "Logistic regression", "SVM (RBF)", "Random forest", "MLP (16-16)"]],
      [3.8, 1.6], 15, hl=1)
text(s, 7.4, 5.2, 5.4, 1.6, [f"Ablation (n = 500): re-uploading (L ≥ 2) and input scaling matter most. "
                           f"Adding the ZZ feature map lowers accuracy ({ab['  with ZZ feature map']:.3f} vs "
                           f"{ab['Re-uploading L=3, ring, scaling (ours)']:.3f})."], 14, MUTED)

# 7 — prediction drives the detour plan, scored against real traffic
s = header("The prediction drives the detour plan", "Results · end to end")
image(s, FIG / "endtoend.png", 0.4, 2.0, w=7.4)
bullets(s, 8.1, 1.75, 4.8, 5.2, [
    "8 on-ramps|x_i = 1 sends ramp i on the detour. Ramp shares are assumed (one sensor in the data).",
    "Expected delay|J(x) = P·J_busy(x) + (1−P)·J_quiet(x) + penalty·x_ambulance. P comes from the quantum classifier.",
    "Gate|detours are released only when P > 0.5 (classical).",
    f"Scored on real traffic|over {R['endtoend_hours']:,} test hours, plans from the VQC get "
    f"{100 * ee.loc['Re-uploading VQC (ours)', 'share_of_possible']:.1f}% of the possible delay reduction "
    f"(fixed plan {100 * ee.loc['Always plan for busy', 'share_of_possible']:.1f}%, perfect classifier "
    f"{100 * ee.loc['Correct label (perfect classifier)', 'share_of_possible']:.1f}%).",
], 15)

# 8 — QAOA
s = header("Detour plan as a QUBO solved by QAOA", "Results · optimisation")
image(s, FIG / "qaoa.png", 0.5, 1.6, w=6.6)
rows = [["Method", "⟨r⟩", "best-plan gap", "ms"]]
for m, r in qa.iterrows():
    rows.append([m, "—" if pd.isna(r.get("approx_ratio")) else f"{r['approx_ratio']:.3f}",
                 "—" if pd.isna(r.get("gap")) else f"{100 * r['gap']:.2f}%", "—" if pd.isna(r.get("ms")) else f"{r['ms']:.1f}"])
table(s, 7.4, 1.6, 5.5, rows, [2.4, 0.9, 1.2, 1.0], 13)
text(s, 0.6, 5.35, 12.1, 1.6, ["30 busy test hours, each with its own P and most with an ambulance ramp. x_i = (1 − Z_i)/2 maps J to an Ising Hamiltonian H_C.",
                          "QAOA lifts the approximation ratio well above random guessing and its best sampled plan is near-optimal, but "
                          "exact search over 256 plans takes under a millisecond. At this size classical solvers win."], 15, MUTED)

# 9 — hardware noise
s = header("Both quantum modules on a noisy device", "Results · deployment")
image(s, FIG / "noise.png", 1.2, 1.5, w=10.9)
bullets(s, 0.6, 4.95, 12.1, 1.9, [
    "Simulated|gate errors (0.025% / 0.75%), T1/T2 decay during gates and idle time (150/100 µs), 1.5% readout error; ring-connected qubits.",
    f"Classifier|accuracy {nz.loc[1, 'acc_mitigated']:.3f} at today's noise (noise-free {nz.loc[0, 'acc_raw']:.3f}). Readout correction + rescaling "
    f"cut the error of P from {nz.loc[1, 'prob_err_raw']:.3f} to {nz.loc[1, 'prob_err_mitigated']:.3f}.",
    f"QAOA|{R['qaoa_circuit']['cx']} CNOTs after routing; approximation ratio {qn.loc[0, 'approx_ratio']:.2f} → {qn.loc[1, 'approx_ratio']:.2f}, "
    f"close to random ({qa.loc['Random guess (p=0)', 'approx_ratio']:.2f}).",
], 15)

# 10 — conclusions
s = header("Conclusions", "Summary")
bullets(s, 0.6, 1.7, 7.2, 5.2, [
    f"Prediction|a day-type qubit plus data re-uploading gives a 4-qubit VQC "
    f"{vqc:.3f} accuracy, level with the best classical model ({best_c}: {acc.loc[best_c, 1000]:.3f}).",
    "Feature maps|the ZZ feature map and the quantum kernel did not help on this low-dimensional data.",
    f"End to end|the VQC's prediction drives detour plans that get {100 * ee.loc['Re-uploading VQC (ours)', 'share_of_possible']:.1f}% of the possible delay reduction on real traffic, level with classical models.",
    "Routing|QAOA produces good plans but offers no advantage at 8 ramps. Classical search is exact and sub-millisecond.",
    "Noise|the shallow classifier survives today's noise with mitigation; the deeper QAOA circuit degrades towards random.",
    "Recommendation|keep ingestion, decisions and actuation classical. Run quantum modules as benchmarked research components behind the same interface.",
], 15)
box(s, 8.3, 1.8, 4.5, 3.6, "Future work",
    "• Real-hardware runs (queue latency, readout error)\n• Multi-junction / graph inputs with more qubits\n"
    "• Regional QAOA sub-problems with 50+ variables\n• Real ramp counts for the detour model\n• Accident detection from camera embeddings", SOFTQ, size=16)

prs.save(OUT / "QuantumITS.pptx")
print("saved", OUT / "QuantumITS.pptx", len(prs.slides), "slides")
