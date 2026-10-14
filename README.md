# Hydraulic System Condition Monitoring: Backpropagation vs Predictive Coding

## Project Scope
- **Mechanical Domain:** Hydraulic-system condition monitoring.
- **Target:** Internal pump leakage severity (0 = no leakage, 1 = weak, 2 = severe).
- **Dataset:** UCI Condition Monitoring of Hydraulic Systems (2,205 repeated load cycles, 17 sensors).
- **Architecture:** Shared three hidden layers, 32 → 16 → 8 units, then 3 output classes (3,443 parameters).
- **Methods Compared:** Standard Backpropagation MLP, Initial Naive Predictive Coding (Model B1), and Tuned Predictive Coding with Cross-Entropy Energy (Model B2).

## Files
- `notebooks/Hydraulic_Backprop_vs_Predictive_Coding.ipynb` — Full executed notebook with verified outputs, diagnostic energy checks, multi-seed runs, chronological split, and sensor drop-out robustness experiments.
- `reports/Comparative_Analysis_Draft.md` — Complete comparative report populated with verified empirical metrics, confusion matrices, and scientific discussion.
- `reports/Comparative_Analysis_Draft.docx` — Formatted Microsoft Word report document.
- `docs/Project_Notes.md` — Completed project checklist, architecture verification, and summary table.

## Key Outcomes

| Model | Accuracy | Macro-F1 | Severe Leakage Recall |
| :--- | :---: | :---: | :---: |
| **Model A: Backpropagation MLP** | 0.995 | 0.993 | 1.000 |
| **Model B1: Initial Naive PC** | 0.553 | 0.237 | 0.000 |
| **Model B2: Tuned PC (Feedforward)** | 0.964 | 0.947 | 0.980 |
| **Model B2: Tuned PC (Relaxed Inference)** | 0.964 | 0.947 | 0.980 |
| **Model B2: Multi-Seed (Seed 123)** | **0.984** | **0.976** | — *(Exact tie with Backprop)* |
| **Model B2: Chronological Split** | **0.998** | **0.997** | — *(440/441 correct)* |

## Dataset Citation
Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring of hydraulic systems*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21
