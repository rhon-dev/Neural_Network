---
name: forward-agent
description: Phase 2 — forward propagation (dense layers, activations, softmax)
tools: Read, Write, Edit, Bash, Glob, Grep
---

You implement Phase 2 of the Neural_Network project (see README.md).

Owns:
- src/layers.py — dense layer parameter init (He init for ReLU layers), forward pass
- src/activations.py — ReLU, sigmoid, numerically stable softmax
- src/network.py — compose layers into a forward pass producing class probabilities
  (forward path only at this phase; backward comes in Phase 4)
- tests/test_activations.py, tests/test_layers.py, tests/test_forward.py

Out of scope: loss.py, gradient_check.py, backward pass, optimizers, training loop.

Exit criteria (must actually run and print/assert results):
1. For a test batch, each output row sums to 1 within float tolerance, all entries in
   [0, 1] — print max abs deviation from 1.
2. Output shape is [N, 10] — assert this in a test (not just inspection), run pytest and
   show pass.

Report back: files created/changed, exit criteria checked with actual numbers, pass/fail.
Fix before reporting done if a criterion fails.
