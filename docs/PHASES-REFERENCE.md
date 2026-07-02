# Phases Reference

Quick lookup per phase. Full narrative in README.md §5.

| Phase | Agent | Deliverable | Exit criteria | Tolerance/target |
|---|---|---|---|---|
| 0 — Data | data-agent | `src/data/` loader | shapes `X:[N,784]` `y:[N,10]`, pixels in `[0,1]`, sample batch renders recognizable digits | — |
| 1 — Gradcheck | gradcheck-agent | `src/gradient_check.py` | numerical vs analytic grad on known function (e.g. `f(x)=sum(x^2)`) | rel error < **1e-7** |
| 2 — Forward | forward-agent | `layers.py`, `activations.py`, forward path in `network.py` | output rows sum to 1, entries in `[0,1]`, shape `[N,10]` | float tolerance |
| 3 — Loss | loss-agent | `src/loss.py` | loss ≥0, near-zero when confident-correct, large when confident-wrong; gradient passes gradcheck | rel error < 1e-7 (or 1e-5 if checked through small network) |
| 4 — Backprop | backprop-agent | backward pass, `network.py` wiring | analytic grads match numerical for ALL params, per-parameter | rel error < **1e-5** |
| 5 — Optimizers | optimizer-agent | `src/optimizers.py` (SGD, mini-batch SGD, momentum, Adam) | each optimizer reduces loss on subset; Adam converges faster than plain SGD (shown side by side) | — |
| 6 — Training | training-agent | `src/train.py` | training loss trends down per epoch; validation accuracy | **> 90%** |
| 7 — Eval | eval-agent | `src/evaluate.py`, docs figures, README §8 | test accuracy in target range; loss/acc curves, confusion matrix, misclassifications saved to docs/ | **95–98%** test acc |
| 8 — Conv (stretch) | conv-agent (TBD) | conv + pooling layers | conv grads pass gradcheck; CNN ≥ MLP baseline accuracy | rel error < 1e-5 |

Architecture fixed across all phases: `784 → Dense(128)+ReLU → Dense(64)+ReLU → Dense(10)+Softmax`, categorical cross-entropy loss.

See [AGENT-ROSTER.md](AGENT-ROSTER.md) for ownership/dependency map, [QA-PUSH-GATE.md](QA-PUSH-GATE.md) for what must be true before pushing a phase.
