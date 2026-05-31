#!/usr/bin/env python3
"""External LLM operator — reads a JSON context payload from stdin, calls the
browser-router's OpenAI-compatible endpoint, and writes a flat JSON tensor-edge
array to stdout for direct ingestion by src/plan.rs.

Exit codes:
  0   success (JSON tensor-edge array on stdout)
  1   API / HTTP error
  127 connection failure (browser-router not reachable)
"""

import sys
import json
import argparse

BROWSER_ROUTER_URL = "http://127.0.0.1:8082/v1/chat/completions"
MODEL = "chatgpt-cdp"

# Valid node names the LLM may reference in edge proposals.
VALID_NODES = (
    "State::Goal", "State::Input", "State::Parse", "State::Map",
    "State::Search", "State::Score", "State::Select", "State::Plan",
    "State::Optimize", "State::Execute", "State::Validate",
    "State::Memory", "State::Learn",
    "Control::Allow", "Control::Block", "Control::Retry", "Control::Repair",
    "Control::Commit", "Control::Rollback", "Control::Halt",
    "Control::GateInput", "Control::GateExecution", "Control::GateReceipt",
    "Control::GateMemory", "Control::GateLearn", "Control::ChooseBest",
    "Event::FactArrived", "Event::InputAccepted", "Event::ParseOk",
    "Event::ParseErr", "Event::MapReady", "Event::CandidateFound",
    "Event::ScoreReady", "Event::TopKSelected", "Event::PlanReady",
    "Event::OptimizeReady", "Event::ExecuteStarted", "Event::ExecuteFinished",
    "Event::ReceiptAttached", "Event::ReceiptAccepted", "Event::ReceiptRejected",
    "Event::HashNonzero", "Event::MemoryWritten", "Event::LearnUpdated",
)

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

PROMPT_TEMPLATES = {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantale LLM operator")
    parser.add_argument("--template", required=True, choices=list(PROMPT_TEMPLATES))
    args = parser.parse_args()

    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"stdin JSON parse error: {exc}\n")
        sys.exit(1)

    context = input_data.get("context", "")
    node_list = "\n".join(f"  {n}" for n in VALID_NODES)
    prompt = PROMPT_TEMPLATES[args.template].format(context=context, nodes=node_list)

    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
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
        sys.stderr.write(f"LLM API HTTP error {exc.code}: {exc.reason}\n")
        sys.exit(1)
    except urllib.error.URLError as exc:
        sys.stderr.write(f"browser-router connection failed: {exc.reason}\n")
        sys.exit(127)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"unexpected response shape: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
