# Alignment Notes After Publishability Revision

This note tracks cross-document wording updates suggested after revising `write-up/paper.tex`.

## 1) Presentation alignment (`checkpoint-2/presentation.tex`)

### A. Exact-solver framing

- Current slide wording suggests large instances require only heuristics/metaheuristics.
- Example location: `checkpoint-2/presentation.tex:376`.
- Recommended wording update:
  - Replace deterministic claim with complementary framing:
  - "Recent exact methods scale strongly on centralized hardware; this work targets lightweight, low-overhead methods for constrained or time-critical deployments."

### B. Hybrid GA large-tier claim range

- Current slide highlights "36--54% better on large instances".
- Example location: `checkpoint-2/presentation.tex:2567`.
- Paper now reports balanced-tier value as 37.70% and mixed small-tier behavior.
- Recommended wording update:
  - "Strong gains at medium/large scale; balanced large-tier improvement = 37.70%."

### C. Statistical-test wording

- Slides mention Wilcoxon test at alpha = 0.05.
- Example location: `checkpoint-2/presentation.tex:2667`.
- Paper now explicitly states one-sided alternative `original > modified`.
- Recommended wording update:
  - Add "one-sided" and interpretation:
  - "Significant result implies modified objective is lower than original."

## 2) Code-doc alignment (`codes/README.md`)

### A. Runtime budget rule

- The paper now explicitly states GA budget rule from implementation:
  - 5 s if max(W,T) <= 30,
  - 20 s if max(W,T) <= 120,
  - 60 s otherwise.
- This is already implemented in `codes/ga_original.py` and applied via `codes/experiment_runner.py`.
- Optional README improvement:
  - Add this exact rule under "Commands" or "Reproducibility Contract".

### B. Reproducibility command parity

- Paper now uses:
  - `cd codes`
  - `pip install -r requirements.txt`
  - `python run_all.py`
- README already supports equivalent flow; no mandatory change needed.

## 3) Citation-policy alignment

- Paper removed non-peer-reviewed core references previously used for foundational claims.
- If slides mention those web-style references, replace with peer-reviewed anchors used in paper bibliography (e.g., Ahuja 2007, Andersen 2022, Bertsimas 2025).
