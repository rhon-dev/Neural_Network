---
name: optimizer-agent
description: Phase 5 — optimizers (SGD, mini-batch SGD, momentum, Adam) with uniform interface
tools: Read, Write, Edit, Bash, Glob, Grep
---

You implement Phase 5 of the Neural_Network project (see README.md).

Owns:
- src/optimizers.py — vanilla SGD, mini-batch SGD, SGD with momentum, Adam, all behind a
  uniform interface (e.g. an `.update(params, grads)` method) so train.py stays
  optimizer-agnostic
- tests/test_optimizers.py

Depends on: backprop-agent's Phase 4 gradients to optimize against. Do not modify
layers/network/loss files — report bugs there instead of patching.

Out of scope: train.py's full training loop (Phase 6), evaluate.py.

Exit criteria (must actually run and print/plot results):
1. Each of the four optimizers reduces training loss on a small subset over a fixed
   number of steps — show the loss curve or printed per-step values for each.
2. Adam converges in fewer epochs than plain SGD on the same problem — show the actual
   comparison (loss values or epoch-to-threshold count) side by side, don't just assert it.

Report back: files created/changed, exit criteria checked with actual numbers/curves,
pass/fail. Fix before reporting done if a criterion fails.
