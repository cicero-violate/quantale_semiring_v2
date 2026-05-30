# Usage

## Prerequisites

- Rust toolchain (2024 edition)
- CUDA 12.x with `nvrtc`
- Python 3 (stdlib only — no extra packages required)
- [browser-router](../browser-router) running on `http://127.0.0.1:8082`

Start the browser-router from its directory before running the agent:

```
cargo run
```

## Running the agent loop

From this directory:

```
cargo run --release
```

The agent loop will:

1. Seed an ingress candidate into the GPU matrix.
2. Tick the quantale closure on the GPU each iteration.
3. Execute the operator mapped to the frontier node (`State::Plan`, `Control::Repair`, etc.).
4. Inject the exit-code feedback weight back into VRAM.
5. Feed the operator's stdout as context into the next iteration.
6. Halt when the matrix reaches a terminal state or `max_ticks` (default 64) is exceeded.

Trace output is appended to `quantale.tlog` (JSONL).

## LLM operators

Two nodes are wired to the external LLM via `assets/call_llm.py`:

| Node | Template | Purpose |
|---|---|---|
| `State::Plan` | `plan` | Generate a structured execution plan from context |
| `Control::Repair` | `repair` | Produce a rollback/retry directive after a failure |

The script sends requests to the browser-router's OpenAI-compatible endpoint
(`POST /v1/chat/completions`) and returns the response text on stdout. Exit codes:

- `0` — success; stdout payload propagates as context to the next step
- `1` — API or HTTP error; GPU down-weights the node and reroutes
- `127` — browser-router unreachable; treated as infrastructure failure

To swap the LLM backend (local Ollama, OpenAI, etc.) edit `BROWSER_ROUTER_URL`
and `MODEL` at the top of `assets/call_llm.py` — no Rust changes required.

## Adding operators

Register any CLI command as a node operator in `assets/operators.json`:

```json
{
  "node_name": "State::Validate",
  "executable": "cargo",
  "static_args": ["test", "--release"],
  "input_mapping": { "stdin_source": null }
}
```

For operators that need the full JSON context payload on stdin (e.g. LLM wrappers):

```json
{
  "node_name": "State::Plan",
  "executable": "python3",
  "static_args": ["assets/call_llm.py", "--template", "plan"],
  "input_mapping": { "stdin_mode": "json" }
}
```

`stdin_mode: "json"` writes the entire current payload as a JSON object to stdin.
`stdin_source: "<field>"` writes only the named field as a plain string.

## Running tests

```
cargo test
```

Tests that require CUDA hardware are in the main binary path; unit and
integration tests (`--lib`, `--test release_validation`) run without a GPU.
