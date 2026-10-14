# Hydraulic System Condition Monitoring: Backpropagation vs Predictive Coding

## Project scope
- Mechanical domain: hydraulic-system condition monitoring.
- Target: internal pump leakage severity (0=no leakage, 1=weak, 2=severe).
- Dataset: UCI Condition Monitoring of Hydraulic Systems.
- Architecture: three hidden layers, 32 → 16 → 8 units, then 3 output classes.
- Methods: standard backpropagation and an explicit predictive-coding-style implementation.

## Files
- `notebooks/Hydraulic_Backprop_vs_Predictive_Coding.ipynb` — notebook for Colab/Jupyter.
- `reports/Comparative_Analysis_Draft.md` — report scaffold; fill in actual metrics after execution.
- `reports/Comparative_Analysis_Draft.docx` — editable Word report scaffold.
- `docs/Project_Notes.md` — design decisions and checklist.

## Run
1. Open the notebook in Google Colab.
2. Run all cells. It attempts to download the UCI archive automatically.
3. If download fails, get the archive from https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems and extract it into `/content/hydraulic_data/`.
4. Use actual outputs to complete the report.

## Important cautions
- Do not invent or copy metrics from another run.
- PC total energy includes hidden-layer prediction errors, unlike backpropagation cross-entropy.
- Random splitting is a teaching baseline and may overstate generalization.
- The dataset is from a lab rig, not field validation.

## Dataset citation
Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring of hydraulic systems*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21
