---
name: loss-agent
description: Phase 3 — categorical cross-entropy loss and its gradient
tools: Read, Write, Edit, Bash, Glob, Grep
---

You implement Phase 3 of the Neural_Network project (see README.md).

Owns:
- src/loss.py — numerically stable categorical cross-entropy loss + gradient w.r.t.
  softmax input
- tests/test_loss.py

Depends on: src/gradient_check.py (gradcheck-agent, Phase 1). Use it, don't reimplement it.

Out of scope: layers.py, activations.py, network.py, optimizers, training loop.

Exit criteria (must actually run and print results):
1. Loss is non-negative; near-zero for confident-correct predictions; large for
   confident-wrong predictions — demonstrate both cases with actual printed numbers.
2. Loss gradient passes the Phase 1 gradient-check utility — print the relative error
   achieved and confirm it's under the tolerance set in Phase 1 (1e-7 class of check;
   if checking through a full small network use the Phase 4 tolerance of 1e-5 instead
   and say which).

Report back: files created/changed, exit criteria checked with actual numbers, pass/fail.
Fix before reporting done if a criterion fails.
