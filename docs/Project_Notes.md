# Project Notes and Checklist

## Locked scope
- Hydraulic system condition monitoring.
- Internal pump leakage target: 0=no leakage, 1=weak, 2=severe.
- Three hidden layers: 32, 16, 8 units; three output classes.
- Both methods use the same split and sensor-derived features.
- Metrics: accuracy, macro-F1, class-wise precision/recall, confusion matrices.
- PC requirements: layer predictions, prediction errors, latent-state updates, layer energies, total energy.

## Before submission
- [ ] Run every notebook cell from a clean runtime.
- [ ] Verify which sensor files were read and the printed feature count.
- [ ] Record actual metrics and include the generated confusion matrices.
- [ ] Explain the PC objective and update schedule accurately.
- [ ] State that PC energy is not directly comparable to backprop cross-entropy.
- [ ] Discuss random-split generalization limits.
- [ ] Cite the UCI dataset.
- [ ] Export the completed report to Google Docs/PDF if required.
