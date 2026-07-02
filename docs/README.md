# docs/

Agent-orchestration and process guides for this project. Start here.

| File | What it's for |
| --- | --- |
| [AGENT-ROSTER.md](AGENT-ROSTER.md) | Which subagent owns which phase and files |
| [PHASES-REFERENCE.md](PHASES-REFERENCE.md) | Quick lookup of exit criteria/tolerances per phase |
| [DEVELOPMENT-FLOW.md](DEVELOPMENT-FLOW.md) | Sequence and loop for invoking phase agents |
| [QA-PUSH-GATE.md](QA-PUSH-GATE.md) | Checklist before marking a phase done / pushing |
| [CODEGUIDE.md](CODEGUIDE.md) | How to move fast here without breaking the gate discipline |

For project architecture, requirements, and the full phase narrative, see the top-level
[README.md](../README.md).

This directory also holds generated figures (`sample_batch.png`, training curves,
confusion matrix — added as phases produce them) and per-phase agent definitions live in
`.claude/agents/`, not here.
