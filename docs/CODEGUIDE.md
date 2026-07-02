# Intern Vibe-Coding Guide

For anyone (human or agent) new to this repo who wants to move fast without breaking the
phase-gate discipline this project depends on.

## The core idea

This project is phase-gated on purpose (README.md §5): each phase's correctness is
verified numerically before the next phase builds on it. "Vibe coding" — writing plausible
code and moving on — is exactly what this structure exists to prevent. Speed comes from
narrow scope per agent, not from skipping verification.

## Rules of thumb

1. **Know which phase you're in.** Check [PHASES-REFERENCE.md](PHASES-REFERENCE.md)
   before touching code. If you're not sure which files you're allowed to touch, check
   [AGENT-ROSTER.md](AGENT-ROSTER.md).
2. **Never eyeball a gradient check.** "Looks about right" is not a pass. Print the
   relative error number and compare to the stated tolerance (1e-7 or 1e-5 depending on
   phase).
3. **If you find a bug outside your lane, don't fix it.** Report file, function, what's
   wrong. Fixing someone else's owned file breaks the ownership model and hides where the
   bug actually was introduced.
4. **Don't lower the bar to make a number pass.** If val accuracy is 88% instead of >90%,
   the fix is tuning (learning rate, epochs, architecture within spec) — not editing the
   threshold or the README's stated target.
5. **Tests are not optional busywork.** Each phase has a `tests/test_*.py`. Write it
   before declaring the phase done, and run it — don't just write code that "should" pass.
6. **Small batches.** One phase, one agent, one commit (or a tight few). Don't let Phase 4
   work bleed into Phase 5 files.
7. **When stuck, ask — don't guess and ship.** Ambiguity in a phase spec gets resolved by
   asking, per [DEVELOPMENT-FLOW.md](DEVELOPMENT-FLOW.md) §4, not by picking whatever's
   easiest and hoping it's right.

## Common failure modes to avoid

- Skipping the numeric gradient check because "the math looks right by inspection."
- Editing a file outside your owned scope because "it was a quick fix."
- Reporting a phase done before actually running the exit-criteria script.
- Silently reducing a target (e.g. "close enough to 90%") instead of iterating or
  escalating.

See [QA-PUSH-GATE.md](QA-PUSH-GATE.md) for the concrete checklist to run before calling
anything done.
