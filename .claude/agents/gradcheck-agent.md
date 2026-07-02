---
name: gradcheck-agent
description: Phase 1 — finite-difference numerical gradient checking utility
tools: Read, Write, Edit, Bash, Glob, Grep
---

You implement Phase 1 of the Neural_Network project (see README.md).

Owns:
- src/gradient_check.py — finite-difference numerical gradient estimator, relative-error
  comparison metric between analytic and numerical gradients, tolerance constants
- tests/test_gradient_check.py

Out of scope: any other src/ module. This is a standalone tooling module used by later
phases (loss, backprop) — do not implement network/loss code here, only the checking
utility itself and its self-test.

Exit criteria (must actually run and print results):
1. On a known closed-form test function (e.g. f(x) = sum(x^2), grad = 2x), relative error
   between numerical and analytic gradient is below 1e-7 — print the actual value achieved.

Report back: files created/changed, exit criteria checked with actual printed numbers,
pass/fail. If the criterion fails, fix (step size, error formula) before reporting done.
