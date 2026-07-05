# Agent Roster

Each phase (README.md §5) has one owning subagent. Agent defs live in `.claude/agents/`.

| Agent | Phase | Owns | Depends on |
|---|---|---|---|
| `data-agent` | 0 | `src/data/`, `requirements.txt`, `.gitignore`, `tests/test_data.py` | — |
| `gradcheck-agent` | 1 | `src/gradient_check.py`, `tests/test_gradient_check.py` | — |
| `forward-agent` | 2 | `src/layers.py`, `src/activations.py`, `src/network.py` (forward path), `tests/test_activations.py`, `tests/test_layers.py`, `tests/test_forward.py` | — |
| `loss-agent` | 3 | `src/loss.py`, `tests/test_loss.py` | gradcheck-agent |
| `backprop-agent` | 4 | backward pass in `src/layers.py`/`src/activations.py`/`src/network.py`, `tests/test_backprop.py` | forward-agent, loss-agent, gradcheck-agent |
| `optimizer-agent` | 5 | `src/optimizers.py`, `tests/test_optimizers.py` | backprop-agent |
| `training-agent` | 6 | `src/train.py`, `tests/test_train.py` | data, forward, loss, backprop, optimizer agents |
| `eval-agent` | 7 | `src/evaluate.py`, `docs/` figures, README §8 Results | training-agent checkpoint |
| (no agent — implemented from main thread) | 8 (stretch) | `src/conv.py`, `src/pooling.py`, `src/cnn.py`, `src/train_cnn.py`, matching tests | eval-agent baseline |

## Rules every agent follows

- **File ownership is strict.** Each agent edits only files listed as "Owns." If it hits a bug in a file it doesn't own, it reports the bug (file, function, what's wrong) instead of patching — the orchestrator re-invokes the owning agent.
- **Exit criteria must be run, not assumed.** Every agent def specifies numeric exit criteria (relative error thresholds, accuracy targets, shape/range checks). Agents print actual numbers and do not report done on a failing gate.
- **Tools available to every phase agent:** Read, Write, Edit, Bash, Glob, Grep. No Agent/Task spawning — orchestration is done by the main thread, not by phase agents.

See [PHASES-REFERENCE.md](PHASES-REFERENCE.md) for full per-phase spec and [DEVELOPMENT-FLOW.md](DEVELOPMENT-FLOW.md) for how these agents get invoked in sequence.
