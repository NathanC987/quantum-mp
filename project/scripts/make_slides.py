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
qe = pd.DataFrame(R["qec"]).set_index("p")
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


def box(s, x, y, w, h, title, body, fill):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill; sh.line.fill.background(); sh.adjustments[0] = 0.12
    tf = sh.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = title; p.font.bold = True; p.font.size = Pt(14); p.font.color.rgb = INK
    p2 = tf.add_paragraph(); p2.text = body; p2.font.size = Pt(13); p2.font.color.rgb = MUTED
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

# 1 — title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0x1E, 0x10, 0x3F); bg.line.fill.background()
text(s, 0.9, 1.6, 11.5, 0.5, "CASE STUDY 5 · NATIONAL INTELLIGENT TRANSPORTATION SYSTEM", 14, RGBColor(0xC4, 0xB5, 0xFD), True)
text(s, 0.9, 2.2, 11.5, 2, "QuantumITS: A Hybrid Quantum–Classical Pipeline for Real-Time Congestion Prediction and Route Optimisation",
     38, BG, True)
text(s, 0.9, 4.5, 11.5, 1.2, ["Data re-uploading VQC · QAOA route planning · Shor-code protection",
                              "Real I-94 traffic + weather data (UCI, 48k hourly records)"], 18, RGBColor(0xDD, 0xD6, 0xFE))
text(s, 0.9, 5.7, 11.5, 0.5, "Rushikesh Shembade  ·  Nathan Cherian", 18, BG, True)
text(s, 0.9, 6.2, 11.5, 0.5, "Department of Computer Engineering, VES. Institute of Technology, Mumbai  ·  Quantum Machine Learning", 14,
     RGBColor(0xA7, 0x8B, 0xFA))

# 2 — problem
s = header("The problem: four decisions, one second", "Motivation")
bullets(s, 0.6, 1.7, 6.3, 5, [
    "Platform|signals, cameras, GPS, emergency vehicles, weather stations and autonomous cars in one system.",
    "Tasks|predict traffic flow, optimise signal timing, detect accidents, recommend alternate routes.",
    "Constraint|every prediction must arrive within 1 second.",
    "Our question|which stages actually benefit from a quantum or hybrid approach, and which must stay classical?",
])
box(s, 7.3, 1.8, 5.4, 1.3, "Prediction → quantum ML", "Is this junction about to be congested? (variational quantum classifier)", SOFTQ)
box(s, 7.3, 3.3, 5.4, 1.3, "Routing → quantum optimisation", "Which traffic streams should be diverted? (QUBO + QAOA)", SOFTQ)
box(s, 7.3, 4.8, 5.4, 1.3, "Reliability → error correction", "Can the quantum output be trusted on noisy hardware? (Shor code)", SOFTQ)

# 3 — data
d = R["dataset"]
s = header("Data and preprocessing", "Dataset")
bullets(s, 0.6, 1.7, 6.4, 5, [
    f"Source|Metro Interstate Traffic Volume (UCI #492 / Kaggle): hourly I-94 volume with weather, 2012–2018.",
    f"Cleaning|removed sensor glitches (9831 mm/h rain, 0 K temperature) and duplicate hours → {d['n_rows']:,} records.",
    f"Label|congested = volume > {d['threshold']:.0f} veh/h (median of the training years).",
    f"Split|chronological. Train 2012–16 ({d['n_train_pool']:,}), test 2017–18 ({d['n_test_period']:,}). No look-ahead.",
])
table(s, 7.3, 1.8, 5.4, [["Feature x_k", "Angle on qubit k"], ["hour of day", "2π·h/24  (cyclic)"],
                         ["weekend / holiday", "0 or π  (basis-like)"], ["temperature", "[0, π]"],
                         ["weather severity 0–4", "[0, π]"]], [2.6, 2.8], 13)
text(s, 7.3, 4.0, 5.4, 2, "Encoding the hour as 2πh/24 makes it periodic: RY(2π) = −I is only a global phase, "
     "so 23:00 sits next to 00:00 with no extra qubit. The weekend flag is basis-encoded as |0⟩ or |1⟩.", 13, MUTED)

# 4 — architecture
s = header("Hybrid architecture: quantum only where it fits", "Method")
stages = [("Sensors", "c"), ("Preprocess", "c"), ("U_enc(x)", "q"), ("U(θ) ×3", "q"), ("Measure ⟨Z₀⟩", "q"),
          ("Decision", "c"), ("QAOA routes", "q"), ("Actuation", "c")]
for i, (n, k) in enumerate(stages):
    box(s, 0.6 + i * 1.53, 1.8, 1.43, 0.9, n, "", SOFTQ if k == "q" else SOFTC)
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
    ["|φ(x)⟩ = U_φ(x)|ψ(x)⟩", "entangled, data-dependent representation (Exp 2). Tested as an ablation"],
    ["U(θ)", "trainable rotations + entanglement. θ learned with Adam on cross-entropy loss"],
    ["measurement → decision", "⟨Z₀⟩ → σ(a⟨Z₀⟩+b) > 0.5 → re-time signals and publish alternate routes"],
], [3.6, 8.5], 14)

# 6 — prediction results
s = header("Prediction: re-uploading VQC matches classical models", "Results · congestion")
image(s, FIG / "learning_curve.png", 0.5, 1.6, w=6.6)
ab = abl["acc"]
table(s, 7.4, 1.7, 5.4, [["Model (n = 1000)", "Accuracy"]] + [
    [m, f"{acc.loc[m, 1000]:.3f}"] for m in ["Re-uploading VQC (ours)", "Standard VQC", "Quantum-kernel SVM",
                                             "Logistic regression", "SVM (RBF)", "Random forest", "MLP (16-16)"]],
      [3.8, 1.6], 13, hl=1)
text(s, 7.4, 4.8, 5.4, 2, [f"Ablation (n = 500): re-uploading (L ≥ 2) and input scaling matter most. "
                           f"Adding the ZZ feature map lowers accuracy ({ab['  with ZZ feature map']:.3f} vs "
                           f"{ab['Re-uploading L=3, ring, scaling (ours)']:.3f})."], 13, MUTED)

# 7 — noise + latency
nz = pd.DataFrame(R["noise"]).set_index("p2")
s = header("Noise and the one-second budget", "Results · deployment")
image(s, FIG / "noise.png", 0.4, 1.7, w=7.6)
image(s, FIG / "latency.png", 8.2, 1.7, w=4.8)
bullets(s, 0.6, 4.4, 12, 2.4, [
    f"Noise|accuracy {nz.loc[0.05, 'acc_raw']:.3f} even at 5% CNOT error (noiseless {R['noise_ideal_acc']:.3f}). "
    f"Noise shrinks ⟨Z₀⟩ ×{nz.loc[0.05, 'shrink']:.2f}, pulling probabilities toward 0.5. A one-number rescaling cuts the probability error "
    f"{nz.loc[0.05, 'prob_err_raw']:.3f} → {nz.loc[0.05, 'prob_err_mitigated']:.3f}.",
    f"Latency|VQC {lat['VQC – exact simulator']['median_ms']:.1f} ms on the simulator, Aer with 1024 shots "
    f"{lat['VQC – Qiskit Aer, 1024 shots']['median_ms']:.1f} ms. Both well under 1 s. Real hardware adds queueing that we did not measure.",
], 14)

# 8 — QAOA
s = header("Route recommendation as a QUBO solved by QAOA", "Results · optimisation")
image(s, FIG / "qaoa.png", 0.5, 1.6, w=6.6)
rows = [["Method", "⟨r⟩", "best-plan gap", "ms"]]
for m, r in qa.iterrows():
    rows.append([m, "—" if pd.isna(r.get("approx_ratio")) else f"{r['approx_ratio']:.3f}",
                 "—" if pd.isna(r.get("gap")) else f"{100 * r['gap']:.2f}%", "—" if pd.isna(r.get("ms")) else f"{r['ms']:.1f}"])
table(s, 7.4, 1.6, 5.5, rows, [2.4, 0.9, 1.2, 1.0], 11)
text(s, 0.6, 5.0, 12, 2, ["J(x) = (D − d·x)²/C_main + (d·x)²/C_alt + τ·d·x + P·x_e on 8 real-demand streams (30 instances). "
                          "x_i = (1 − Z_i)/2 maps J to an Ising Hamiltonian H_C.",
                          "QAOA lifts the approximation ratio well above random guessing and its sampled plans reach the optimum, but "
                          "exact enumeration of 256 plans takes under a millisecond. At this size classical solvers win."], 14, MUTED)

# 9 — QEC
s = header("Protecting the quantum output with error correction", "Results · reliability")
image(s, FIG / "qec.png", 0.6, 1.6, w=5.8)
table(s, 6.9, 1.7, 6.0, [["Code", "qubits", "protects", "p_L at p = 1%"],
                         ["Unprotected", 1, "—", f"{qe.loc[0.01, 'unprotected']:.4f}"],
                         ["Bit-flip (Exp 7)", 3, "X", f"{qe.loc[0.01, 'bit_flip']:.4f}"],
                         ["Phase-flip (Exp 7)", 3, "Z", f"{qe.loc[0.01, 'phase_flip']:.4f}"],
                         ["Shor (Exp 8)", 9, "X, Y, Z", f"{qe.loc[0.01, 'shor']:.4f}"]], [2.0, 0.9, 1.3, 1.8], 13, hl=4)
text(s, 6.9, 3.9, 6.0, 3, ["Under depolarising noise the 3-qubit codes do worse than no code at all: each lets one error type through and offers three targets.",
                           "Shor's code cuts the logical error roughly to p², at a cost of 9× qubits (36 for the 4-qubit VQC)."], 14, MUTED)

# 10 — conclusions
s = header("Conclusions", "Summary")
bullets(s, 0.6, 1.7, 7.2, 5.2, [
    f"Prediction|a day-type qubit plus data re-uploading lifts a 4-qubit VQC from ~0.66 (Exp 5) to "
    f"{vqc:.3f} accuracy, level with the best classical model ({best_c}: {acc.loc[best_c, 1000]:.3f}).",
    "Feature maps|the ZZ feature map and the quantum kernel did not help on this low-dimensional data.",
    "Routing|QAOA produces good plans but offers no advantage at 8 variables. Classical solvers are exact and sub-millisecond.",
    "Reliability|only Shor's code beats an unprotected qubit under realistic noise.",
    "Recommendation|keep ingestion, decisions and actuation classical. Run quantum modules as benchmarked research components behind the same interface.",
], 15)
box(s, 8.3, 1.8, 4.5, 4.6, "Future work",
    "• Real-hardware runs (queue latency, readout error)\n• Multi-junction / graph inputs with more qubits\n"
    "• Regional QAOA sub-problems with 50+ variables\n• Accident detection from camera embeddings\n• Fault-tolerant cost estimates", SOFTQ)

prs.save(OUT / "QuantumITS.pptx")
print("saved", OUT / "QuantumITS.pptx", len(prs.slides), "slides")
