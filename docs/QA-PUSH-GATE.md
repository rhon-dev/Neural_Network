# QA / Push Gate

Checklist before merging or pushing any phase's work. Mirrors the "exit criteria" language
already baked into each agent def — this file is the orchestrator-side enforcement of it.

## Before marking a phase done

- [ ] Exit criteria for the phase (see [PHASES-REFERENCE.md](PHASES-REFERENCE.md)) were
      actually **run**, not assumed. Numbers were printed, not eyeballed.
- [ ] `pytest` passes for the phase's new test file(s) and the full suite still passes
      (no regression in earlier phases' tests).
- [ ] Agent touched only files it owns (see [AGENT-ROSTER.md](AGENT-ROSTER.md)). Any
      cross-file bug was reported, not silently patched.
- [ ] No lowered bar: if a target wasn't hit (e.g. val accuracy ≤ 90%, test acc outside
      95–98%), the agent iterated (tuning within its own files) or reported the real
      number — it did not soften the stated target to make it "pass."

## Before pushing / committing

- [ ] `git status` reviewed — no stray files (checkpoints, `data/`, `.venv/`) staged.
- [ ] Commit scoped to one phase (or a clearly-related slice of one).
- [ ] README.md §5 phase status still accurate if the phase changes what's documented
      there.

## Numeric gates by phase (quick reference)

| Phase | Gate |
|---|---|
| 1 | gradcheck self-test rel error < 1e-7 |
| 2 | softmax rows sum to 1, shape `[N,10]` |
| 3 | loss gradient rel error < 1e-7 (or 1e-5 through small network) |
| 4 | ALL param grads rel error < 1e-5 |
| 6 | val accuracy > 90% |
| 7 | test accuracy in 95–98% |
| 8 | conv grads rel error < 1e-5; CNN ≥ MLP baseline |

If a gate fails, do not proceed to the next phase's agent. Fix or report, per
[DEVELOPMENT-FLOW.md](DEVELOPMENT-FLOW.md) §2.
