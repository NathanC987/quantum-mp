# Case Study 5: National Intelligent Transportation System (QML coursework)

| Folder | Contents |
|---|---|
| `Exp1…Exp6 *.pdf` | Your experiments 1–6 |
| `experiments/` | **Exp 7** (3-qubit bit-flip and phase-flip codes) and **Exp 8** (Shor 9-qubit code), as Colab-ready `.ipynb` notebooks with outputs and as `.pdf`. `build_exp*.py` regenerates them. |
| `project/` | **QuantumITS**: the full hybrid quantum–classical project (library, benchmarks, web dashboard). See `project/README.md` |
| `slides/QuantumITS.pptx` | 10-slide presentation |
| `paper/` | IEEE conference paper (`main.tex` + figures + `numbers.tex`, auto-generated from the results) |
| `data/` | Metro Interstate Traffic Volume dataset (UCI #492) |

**Paper PDF:** `paper/main.pdf`. To rebuild: `docker run --rm -v "$PWD/paper":/work -w /work texlive/texlive:latest pdflatex main.tex` (run twice), or upload `paper/` to Overleaf.

**Run the dashboard:**
```bash
cd project && ../.venv/bin/uvicorn app.server:app --port 8000   # http://localhost:8000
```
