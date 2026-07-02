# Development Flow

How work moves through this repo, phase by phase, agent by agent.

## 1. Sequence

Phases are strictly ordered (0 → 8). A phase's agent depends on prior phases' code being
present and gate-passed — see dependency column in [AGENT-ROSTER.md](AGENT-ROSTER.md).
Don't start phase N+1's agent until phase N's exit criteria have been run and printed
green.

```
data → gradcheck → forward → loss → backprop → optimizer → training → eval → (conv, stretch)
```

gradcheck-agent (Phase 1) is a dependency-free tooling module; it can run in parallel with
data-agent (Phase 0) if desired since neither touches the other's files.

## 2. Per-phase loop

1. Orchestrator (main thread / you) invokes the phase's agent via `Agent` tool with
   `subagent_type` matching the agent name in `.claude/agents/`.
2. Agent implements only files it owns, writes tests, runs its exit criteria, and prints
   actual numbers (not "looks right").
3. Agent reports back: files changed, exit criteria values, pass/fail.
4. If a criterion fails, the agent fixes it (within its own files) before reporting done.
   If the bug is in a file it doesn't own, it reports the bug precisely and stops —
   orchestrator re-invokes the owning agent.
5. Orchestrator verifies the report (re-run pytest / spot check) before moving to the
   next phase. See [QA-PUSH-GATE.md](QA-PUSH-GATE.md).

## 3. Commit discipline

- One phase's work = one logical commit (or a few, if the phase is large). Don't mix
  phases in a commit.
- Never commit with a failing exit criterion.

## 4. When something's unclear

If a phase spec seems ambiguous or an agent hits a design decision not covered by
README.md §5, stop and ask rather than guessing — the roadmap is deliberately
phase-gated so ambiguity should be resolved before code lands, not patched after.
