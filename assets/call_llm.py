#!/usr/bin/env python3
"""External LLM operator — reads a JSON context payload from stdin, calls the
browser-router's OpenAI-compatible endpoint, and writes a flat JSON tensor-edge
array to stdout for direct ingestion by src/plan.rs.

Exit codes:
  0   success (JSON tensor-edge array on stdout)
  1   API / HTTP error
  127 connection failure (browser-router not reachable)

Configuration (all via environment variables):
  BROWSER_ROUTER_URL       default http://127.0.0.1:8090/v1/chat/completions
  BROWSER_ROUTER_MODEL     default chatgpt-cdp
  BROWSER_ROUTER_PROVIDER  default chatgpt_private
  BROWSER_ROUTER_TARGET_URL default https://chatgpt.com/
  QUANTALE_TOPOLOGY        path to topology.json, default assets/topology.json
  QUANTALE_TEMPLATES       path to call_llm_templates.json, default assets/call_llm_templates.json
"""

import sys
import json
import argparse
import os
import pathlib

BROWSER_ROUTER_URL = os.environ.get(
    "BROWSER_ROUTER_URL", "http://127.0.0.1:8090/v1/chat/completions"
)
MODEL             = os.environ.get("BROWSER_ROUTER_MODEL", "chatgpt-cdp")
BROWSER_PROVIDER  = os.environ.get("BROWSER_ROUTER_PROVIDER", "chatgpt_private")
BROWSER_TARGET    = os.environ.get("BROWSER_ROUTER_TARGET_URL", "https://chatgpt.com/")
TOPOLOGY_PATH     = os.environ.get("QUANTALE_TOPOLOGY", "assets/topology.json")
TEMPLATES_PATH    = os.environ.get("QUANTALE_TEMPLATES", "assets/call_llm_templates.json")

_EDGE_SCHEMA = """\
Output ONLY a JSON array of tensor edge objects — no prose, no markdown fences.
Each object must have exactly these keys:
  "from"       : a node name from the valid set below
  "to"         : a node name from the valid set below
  "confidence" : a float in [0.0, 1.0] for correctness/confidence
  "cost"       : a nonnegative float for compute/time cost
  "safety"     : a float in [0.0, 1.0] for security/safety

Valid node names:
{nodes}

Example output:
[
  {{"from": "State::Plan", "to": "State::Optimize", "confidence": 0.95, "cost": 2.0, "safety": 0.90}},
  {{"from": "State::Optimize", "to": "State::Execute", "confidence": 0.90, "cost": 3.0, "safety": 0.88}}
]"""

_BUILTIN_TEMPLATES = {
    "plan": (
        "You are a neuro-symbolic planning engine embedded in a quantale-matrix agent loop.\n"
        "Your output is compiled directly into GPU matrix weights — it must be machine-readable.\n\n"
        "Execution context:\n{context}\n\n"
        "Propose a sequence of state transitions that will move the agent toward its goal.\n"
        "Choose high weights (0.85–0.99) for steps you are confident in; "
        "lower weights (0.50–0.84) for speculative steps.\n\n"
        + _EDGE_SCHEMA
    ),
    "repair": (
        "You are a repair subsystem for a quantale-matrix agent loop.\n"
        "Your output is compiled directly into GPU matrix weights — it must be machine-readable.\n\n"
        "The system encountered a failure. Execution context:\n{context}\n\n"
        "Propose a recovery path by strengthening edges toward rollback, retry, or repair nodes.\n"
        "Use weights close to 1.0 for the recovery path you recommend most strongly.\n\n"
        + _EDGE_SCHEMA
    ),
}


def load_valid_nodes() -> tuple[str, ...]:
    """Load node names from assets/topology.json at runtime."""
    try:
        data = json.loads(pathlib.Path(TOPOLOGY_PATH).read_text())
        return tuple(n["name"] for n in data.get("nodes", []))
    except Exception as exc:
        sys.stderr.write(f"[call_llm] topology load failed ({TOPOLOGY_PATH}): {exc}\n")
        return ()


def load_templates() -> dict[str, str]:
    """Load prompt templates from assets/call_llm_templates.json if it exists,
    otherwise fall back to the built-in set."""
    try:
        data = json.loads(pathlib.Path(TEMPLATES_PATH).read_text())
        return {k: v + "\n\n" + _EDGE_SCHEMA for k, v in data.items()}
    except FileNotFoundError:
        return _BUILTIN_TEMPLATES
    except Exception as exc:
        sys.stderr.write(f"[call_llm] template load failed ({TEMPLATES_PATH}): {exc}\n")
        return _BUILTIN_TEMPLATES


def main() -> None:
    templates = load_templates()

    parser = argparse.ArgumentParser(description="Quantale LLM operator")
    parser.add_argument("--template", required=True, choices=list(templates))
    args = parser.parse_args()

    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"stdin JSON parse error: {exc}\n")
        sys.exit(1)

    valid_nodes = load_valid_nodes()
    node_list = "\n".join(f"  {n}" for n in valid_nodes) if valid_nodes else "  (topology unavailable)"

    context = input_data.get("context", "")
    prompt = templates[args.template].format(context=context, nodes=node_list)

    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "browser": {
            "provider": BROWSER_PROVIDER,
            "target_url": BROWSER_TARGET,
        },
    }).encode()

    req = urllib.request.Request(
        BROWSER_ROUTER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode())
            content = body["choices"][0]["message"]["content"]
            print(content)
            sys.exit(0)

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace").strip()
        detail = f": {error_body}" if error_body else ""
        sys.stderr.write(f"LLM API HTTP error {exc.code}: {exc.reason}{detail}\n")
        sys.exit(1)
    except urllib.error.URLError as exc:
        sys.stderr.write(f"browser-router connection failed: {exc.reason}\n")
        sys.exit(127)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"unexpected response shape: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
