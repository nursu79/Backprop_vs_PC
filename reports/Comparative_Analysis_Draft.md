# Comparative Analysis Draft
## Backpropagation vs. Predictive Coding for Hydraulic System Condition Monitoring

**Status:** Draft scaffold. Complete the results and conclusion after running the notebook.

## 1. Engineering context and research question

Hydraulic systems transmit power through pressurized fluid. Internal pump leakage can reduce effective flow and system performance. Condition monitoring uses measurements such as pressure, motor power, volume flow, temperature, vibration, and efficiency-related signals to infer component condition.

This project uses the UCI *Condition monitoring of hydraulic systems* experimental test-rig dataset. The project focuses on internal pump leakage, classified as 0 (no leakage), 1 (weak leakage), or 2 (severe leakage).

**Research question:** Under the same three-hidden-layer architecture and held-out split, how do a standard backpropagation MLP and a predictive-coding implementation compare on pump-leakage classification, and how do their learning mechanisms differ?

## 2. Dataset and feature engineering

The raw dataset contains multirate sensor signals organized by cycle. Each sensor cycle is summarized using mean, standard deviation, minimum, maximum, and median. This produces a fixed-length feature vector while discarding temporal ordering and waveform shape. The target comes from the third column of `profile.txt`; target annotations are excluded from model inputs. Standardization is fitted on the training split only.

**Evaluation caveat:** A random stratified split may place highly related operating cycles in both train and test sets. Treat this as a teaching benchmark, not proof of generalization to unseen operating conditions. A condition-aware/group-aware split is a recommended follow-up.

## 3. Shared architecture

Both models use the same structure:

- Input: engineered cycle-level sensor features (the notebook prints the exact number).
- Hidden layer 1: 32 ReLU units.
- Hidden layer 2: 16 ReLU units.
- Hidden layer 3: 8 ReLU units.
- Output: three leakage classes.

This network has **three hidden layers**, as requested. Both methods use the same train/test split and scaled features.

## 4. Model A — standard backpropagation

The MLP minimizes multiclass cross-entropy. Automatic differentiation computes gradients of the output loss through the full network, and Adam updates weights and biases. This is a conventional supervised-learning baseline with mature optimization tools.

Research concerns sometimes raised about backpropagation include gradient transport through successive layers, the relationship between forward and backward weights in standard implementations, and the separation of forward and backward computations. These are motivations in debates about biological plausibility; they do not mean backpropagation is ineffective for engineering tasks.

## 5. Model B — predictive coding

The PC implementation uses three hidden latent states with dimensions 32, 16, and 8. During training, the target is represented as a clamped one-hot state. Each layer predicts the next state, producing errors:

ε_l = z_l − μ_l

The layer energy is:

E_l = 1/2 × mean(||ε_l||²)

The total energy is the sum of the layer energies:

E = Σ_l E_l

For each minibatch, hidden latent states are first iteratively updated to reduce total energy while weights are held fixed. Then each layer's weights are updated from that layer's local energy while inferred states are held fixed. At test time, targets are unavailable, so the notebook uses feedforward initialization to generate output probabilities.

This is one explicit predictive-coding-style formulation, not the only formulation. Other PC models may use different generative graphs, state-update rules, nonlinearities, or learning schedules.

## 6. Results — fill after running the notebook

**Use the actual outputs; do not invent or estimate metrics.**

| Metric | Backpropagation MLP | Predictive Coding MLP |
|---|---:|---:|
| Accuracy | [paste actual result] | [paste actual result] |
| Macro-F1 | [paste actual result] | [paste actual result] |
| No-leakage recall | [paste report] | [paste report] |
| Weak-leakage recall | [paste report] | [paste report] |
| Severe-leakage recall | [paste report] | [paste report] |

Insert the confusion matrices and training curves produced by the notebook. Discuss which classes are confused and whether severe leakage is detected reliably. Do not select a winner based on accuracy alone if class-wise recall or macro-F1 tells a different story.

## 7. Interpretation

The two models share the architecture and split, but optimize different objectives. Backpropagation minimizes output cross-entropy. The PC model minimizes a sum of hidden-layer and output-layer prediction energies using iterative state inference and local parameter updates. This is a comparison of learning procedures, not an experiment with identical objectives.

If backpropagation performs better, possible explanations include direct optimization of the supervised objective and the chosen PC inference/update schedule. If PC performs better, that result still needs to be tested across seeds and alternative splits. Interpret the observed results, not prior expectations.

## 8. Limitations

1. The dataset comes from one experimental test rig, not a representative fleet of field machines.
2. Random splitting can overestimate generalization to unseen operating conditions.
3. Statistical summaries discard temporal order and waveform shape.
4. A single run does not establish a robust ranking.
5. PC performance depends on formulation, latent-state initialization, inference steps, learning rates, and local update rules.
6. PC total energy is not directly comparable to backpropagation cross-entropy because it includes hidden-layer prediction errors.
7. This study does not prove that predictive coding resolves all gradient-transport, weight-symmetry, update-locking, or biological-plausibility concerns.
8. The notebook is an educational experiment, not a validated safety-critical maintenance system.

## 9. Conclusion — complete after reviewing results

Summarize (a) the mechanical task, (b) the strongest measured finding, (c) the learning-mechanism difference, and (d) the most important limitation. Keep conclusions proportional to this dataset and evaluation design.

## Reference

Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring of hydraulic systems* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21
