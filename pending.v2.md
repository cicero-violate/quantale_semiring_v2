# pending.v2.md

## Bugs

**Memory output key mismatch**
`memory.py` emits `{"memory": [...]}` but `call_llm.py` reads `payload.get("context", "")`.
The accumulated memory never reaches the LLM prompt. Fix: either have `memory.py` emit
`{"context": ...}` or have `call_llm.py` check for the `memory` key and format it into
the prompt.

**State::Validate re-entry loop**
Receipt edges route `ReceiptAccepted → HashNonzero → State::Validate` and
`HashNonzero → State::Validate` is also a topology edge. The consumed mask doesn't
prevent re-entry because each path is a different (src, dst) pair. `State::Validate`
runs `cargo test --release` 3 times per cycle as a result. Needs a per-node visit cap
or a receipt.json rule that doesn't loop back after the first acceptance.

**LLM plan fires too late**
`State::Plan` is step 15. By then the consumed mask has already passed the nodes the
LLM's 35 edges target. The plan modifies the matrix but the frontier has moved past it.
The plan weights only take effect on the *next run*. To fix: either move `State::Plan`
earlier in the topology or flush consumed state after a plan is loaded.

**`Control::GateExecution` is a no-op placeholder**
`patch -p1 --batch` on empty stdin exits 0 always. It doesn't actually gate anything.
Needs a real predicate — check for a diff in `current_payload`, or replace with a
script that validates pre-conditions before allowing execution.

---

## Missing operators

**State::Learn**
The topology routes through it every cycle but the slot is empty (exit 127).
Natural implementation: read `state/memory.jsonl`, compute a frequency or
recency score over recent context entries, write a small JSON weight file
(`state/learned_weights.json`) that can be loaded as edge deltas on the next
run. Closes the feedback loop between memory and future routing.

**Control::Commit**
Currently exit 127. Should do something durable: snapshot `quantale.tlog`,
write a run summary to `state/runs/`, or `git commit` the memory file.
One small shell script or Python script, one operators.json entry.

---

## Architectural improvements

**Multi-turn LLM conversation history**
Each `call_llm.py` invocation is a fresh single-turn prompt. For coherent
planning across steps, `State::Plan` should carry a `messages` array through
`current_payload` so the LLM can see its own prior output. The browser-router
already accepts the full messages array.

**`state/` in .gitignore**
`state/memory.jsonl`, `state/runs/`, and `quantale.tlog` are runtime artefacts
and should not be committed. Add `state/` and `*.tlog` to `.gitignore`.

**`State::Validate` scope**
It runs `cargo test --release` — all suites, optimised build, every cycle.
`cargo test --lib` is faster for the validation step and sufficient for
checking algebraic invariants. The release build only makes sense for a
pre-commit gate, not an in-loop validator.

**LLM plan edges are ephemeral**
Plan edges loaded into VRAM are lost when the process exits. If `State::Learn`
wrote them to `state/learned_weights.json` and main.rs loaded that file on
startup (alongside `default_transition_edges()`), the LLM's accumulated
planning decisions would persist and compound across runs.

---

## Cleanup

- `USAGE.md` doesn't mention `memory.py`, `plan.rs`, or `output_mode`
- `pending.md` and `plan.md` / `plan_1.md` at project root are stale, can be deleted
- The `pub use plan::*` glob export in `lib.rs` exposes `compile_llm_plan` and the
  private `PlanEdge` deserialiser at crate root — consider scoping the export
