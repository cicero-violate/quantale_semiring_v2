#!/usr/bin/env python3
"""External LLM operator — reads a JSON context payload from stdin, calls the
browser-router, and writes the reply to stdout for ingestion by the agent loop.

Exit codes:
  0   success
  1   API / HTTP error
  127 connection failure (browser-router not reachable)

Configuration (all via environment variables):
  BROWSER_ROUTER_URL            default http://127.0.0.1:8090/v1/chat/completions
  BROWSER_ROUTER_MODEL          default chatgpt-cdp
  BROWSER_ROUTER_PROVIDER       default chatgpt_private
  BROWSER_ROUTER_TARGET_URL     default https://chatgpt.com/
  BROWSER_ROUTER_GROUP_CHAT_URL if set to a https://chatgpt.com/gg/... URL, all
                                 requests are sent to that group chat room via the
                                 /actions/group-chat endpoint.  The group chat URL
                                 overrides model, provider, and target_url.
  QUANTALE_TOPOLOGY             path to topology.json, default assets/topology.json
  QUANTALE_OPERATORS            path to operators.json, default assets/operators.json
  QUANTALE_TEMPLATES            path to call_llm_templates.json, default assets/call_llm_templates.json
"""

import sys
import json
import argparse
import os
import pathlib

_ROUTER_BASE = os.environ.get("BROWSER_ROUTER_URL", "http://127.0.0.1:8090/v1/chat/completions")
# Derive the base URL (scheme + host + port) from BROWSER_ROUTER_URL so that
# the group-chat action endpoint lives on the same server.
_ROUTER_ORIGIN = "/".join(_ROUTER_BASE.split("/")[:3])  # e.g. http://127.0.0.1:8090

BROWSER_ROUTER_URL    = _ROUTER_BASE
GROUP_CHAT_ACTION_URL = _ROUTER_ORIGIN + "/actions/group-chat"
GROUP_CHAT_URL        = os.environ.get("BROWSER_ROUTER_GROUP_CHAT_URL", "").strip()
MODEL                 = os.environ.get("BROWSER_ROUTER_MODEL", "chatgpt-cdp")
BROWSER_PROVIDER      = os.environ.get("BROWSER_ROUTER_PROVIDER", "chatgpt_private")
BROWSER_TARGET        = os.environ.get("BROWSER_ROUTER_TARGET_URL", "https://chatgpt.com/")
ASSET_DIR             = pathlib.Path(__file__).resolve().parent


def asset_path(env_name: str, filename: str) -> pathlib.Path:
    configured = os.environ.get(env_name)
    return pathlib.Path(configured) if configured else ASSET_DIR / filename


TOPOLOGY_PATH     = asset_path("QUANTALE_TOPOLOGY", "topology.json")
OPERATORS_PATH    = asset_path("QUANTALE_OPERATORS", "operators.json")
TEMPLATES_PATH    = asset_path("QUANTALE_TEMPLATES", "call_llm_templates.json")

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

Valid topology transitions:
{transitions}

Available JIT execution operators, loaded from operators.json:
{jit_operators}

Example output:
{example_edges}"""

_BUILTIN_TEMPLATES = {
    "plan": (
        "You are a neuro-symbolic planning engine embedded in a quantale-matrix agent loop.\n"
        "Your output is compiled directly into GPU tensor weights, so it must be data-only JSON.\n\n"
        "Prior execution context summary:\n{context}\n\n"
        "Propose an ordered execution chain by emitting tensor edges that connect valid topology nodes.\n"
        "A chain is represented only by consecutive edge objects: A -> B, B -> C, C -> D.\n"
        "Prefer edges from the valid topology transitions list; add new edges only when the context justifies them.\n"
        "For JIT-capable work, prefer adjacent jit_cuda execution nodes when their declared effects form a data dependency.\n"
        "Do not invent node names, slot names, operators, kernels, Rust symbols, or CUDA code.\n"
        "Do not output a separate chains object; the edge array is the structured chain.\n\n"
        + _EDGE_SCHEMA
    ),
    "repair": (
        "You are a repair subsystem for a quantale-matrix agent loop.\n"
        "Your output is compiled directly into GPU tensor weights, so it must be data-only JSON.\n\n"
        "The system encountered a failure. Prior execution context summary:\n{context}\n\n"
        "Propose a recovery execution chain by emitting consecutive tensor edges between valid topology nodes.\n"
        "Prefer edges from the valid topology transitions list; add new edges only when the context justifies them.\n"
        "Choose recovery behavior only from node names that actually appear in the valid topology list.\n"
        "For JIT-capable recovery work, use adjacent jit_cuda execution nodes only when their declared effects form a data dependency.\n"
        "Do not invent node names, slot names, operators, kernels, Rust symbols, or CUDA code.\n\n"
        + _EDGE_SCHEMA
    ),
}

# Templates that emit tensor edge arrays; all others emit task-specific JSON.
_EDGE_SCHEMA_TEMPLATES: frozenset[str] = frozenset({"plan", "repair"})


def load_topology() -> dict:
    """Load topology.json from the configured asset path."""
    return json.loads(TOPOLOGY_PATH.read_text())


def load_valid_nodes() -> tuple[str, ...]:
    """Load node names from assets/topology.json at runtime."""
    try:
        data = load_topology()
        return tuple(n["name"] for n in data.get("nodes", []))
    except Exception as exc:
        sys.stderr.write(f"[call_llm] topology load failed ({TOPOLOGY_PATH}): {exc}\n")
        return ()


def load_transition_summary() -> str:
    """Load legal transition declarations from topology.json."""
    try:
        data = load_topology()
        lines = []
        for edge in data.get("transitions", []):
            src = edge.get("from")
            dst = edge.get("to")
            if not src or not dst:
                continue
            cost = edge.get("cost")
            confidence = edge.get("confidence")
            safety = edge.get("safety")
            lines.append(
                f"  {src} -> {dst} "
                f"(confidence={confidence}, cost={cost}, safety={safety})"
            )
        return "\n".join(lines) if lines else "  (no transitions declared)"
    except Exception as exc:
        sys.stderr.write(f"[call_llm] transition load failed ({TOPOLOGY_PATH}): {exc}\n")
        return "  (topology transitions unavailable)"


def load_jit_operator_summary(valid_nodes: tuple[str, ...]) -> str:
    """Load JIT operator data-flow declarations from assets/operators.json."""
    try:
        data = json.loads(OPERATORS_PATH.read_text())
        valid_node_set = set(valid_nodes)
        lines = []
        for op in data.get("operators", []):
            if op.get("executable") != "jit_cuda":
                continue
            node_name = op.get("node_name")
            if node_name not in valid_node_set:
                continue
            effects = op.get("effects", {})
            reads = ", ".join(effects.get("reads", []))
            writes = ", ".join(effects.get("writes", []))
            lines.append(f"  {node_name}: reads [{reads}] -> writes [{writes}]")
        return "\n".join(lines) if lines else "  (no topology-visible jit_cuda operators declared)"
    except Exception as exc:
        sys.stderr.write(f"[call_llm] operator load failed ({OPERATORS_PATH}): {exc}\n")
        return "  (operator registry unavailable)"


def example_edges() -> str:
    """Return a trading-path example for the plan prompt."""
    example = [
        {"from": "State::Input", "to": "State::MarketFeed", "confidence": 0.97, "cost": 0.03, "safety": 0.97},
        {"from": "State::MarketFeed", "to": "Event::MarketFeedUpdated", "confidence": 0.92, "cost": 0.08, "safety": 0.92},
        {"from": "Event::MarketFeedUpdated", "to": "State::AnalysisPlan", "confidence": 0.91, "cost": 0.09, "safety": 0.91},
    ]
    return json.dumps(example, indent=2)


def edge_object(edge: dict, fallback_confidence: float) -> dict:
    return {
        "from": edge.get("from"),
        "to": edge.get("to"),
        "confidence": edge.get("confidence", fallback_confidence),
        "cost": edge.get("cost", 1.0),
        "safety": edge.get("safety", fallback_confidence),
    }


def load_templates() -> dict[str, str]:
    """Load prompt templates from assets/call_llm_templates.json if it exists,
    otherwise fall back to the built-in set.

    Templates in _EDGE_SCHEMA_TEMPLATES receive the tensor-edge output schema
    appended automatically. All other templates include their own output schema.
    """
    try:
        data = json.loads(TEMPLATES_PATH.read_text())
        result = {}
        for k, v in data.items():
            if k in _EDGE_SCHEMA_TEMPLATES:
                result[k] = v + "\n\n" + _EDGE_SCHEMA
            else:
                result[k] = v
        return result
    except FileNotFoundError:
        return _BUILTIN_TEMPLATES
    except Exception as exc:
        sys.stderr.write(f"[call_llm] template load failed ({TEMPLATES_PATH}): {exc}\n")
        return _BUILTIN_TEMPLATES


def load_asset_json_str(filename: str) -> str:
    """Load an asset JSON file as a compact string; return empty string if missing."""
    path = ASSET_DIR / filename
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return ""
    except Exception as exc:
        sys.stderr.write(f"[call_llm] asset load failed ({path}): {exc}\n")
        return ""


def load_jit_analysis_operator_summary(valid_nodes: tuple[str, ...]) -> str:
    """Load JIT analysis operators (Analysis:: prefix) from assets/operators.json."""
    try:
        data = json.loads(OPERATORS_PATH.read_text())
        valid_node_set = set(valid_nodes)
        lines = []
        for op in data.get("operators", []):
            if op.get("executable") != "jit_cuda":
                continue
            node_name = op.get("node_name", "")
            if not node_name.startswith("Analysis::"):
                continue
            if node_name not in valid_node_set:
                continue
            effects = op.get("effects", {})
            reads = ", ".join(effects.get("reads", []))
            writes = ", ".join(effects.get("writes", []))
            jit_body = op.get("jit_body", "")
            lines.append(f"  {node_name}: reads [{reads}] -> writes [{writes}]  body: {jit_body}")
        return "\n".join(lines) if lines else "  (no topology-visible Analysis:: jit_cuda operators declared)"
    except Exception as exc:
        sys.stderr.write(f"[call_llm] analysis operator load failed: {exc}\n")
        return "  (analysis operator registry unavailable)"


def normalize_context(value) -> str:
    """Unwrap repeated {"context": "..."} envelopes and render compact data."""
    current = value
    for _ in range(8):
        if isinstance(current, dict) and set(current.keys()) == {"context"}:
            current = current["context"]
            continue
        if isinstance(current, str):
            stripped = current.strip()
            try:
                current = json.loads(stripped)
                continue
            except json.JSONDecodeError:
                extracted = extract_json_array(stripped)
                if extracted is not None:
                    current = extracted
                    continue
                return compact_text(stripped)
        break

    if isinstance(current, list):
        return summarize_json_list(current)
    if isinstance(current, (dict, tuple)):
        return json.dumps(current, ensure_ascii=True, separators=(",", ":"))
    return compact_text(str(current))


def extract_json_array(text: str):
    """Parse a JSON array embedded in a larger string, if one is present."""
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def compact_text(text: str, limit: int = 2048) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "...[truncated]"


def summarize_json_list(items: list) -> str:
    """Keep prior JSON arrays visible without recursively flooding the prompt."""
    if not items:
        return "[]"
    if all(isinstance(item, dict) and {"from", "to"} <= set(item) for item in items):
        jit_edges = [
            item
            for item in items
            if str(item.get("from", "")).startswith("Execution::")
            or str(item.get("to", "")).startswith("Execution::")
        ]
        summary = {
            "prior_tensor_edge_count": len(items),
            "first_edge": items[0],
            "last_edge": items[-1],
        }
        if jit_edges:
            summary["prior_execution_edges"] = jit_edges[:8]
        return json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    return json.dumps(items[:16], ensure_ascii=True, separators=(",", ":"))


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
    transitions = load_transition_summary()
    jit_operators = load_jit_operator_summary(valid_nodes)
    jit_analysis_operators = load_jit_analysis_operator_summary(valid_nodes)

    context = normalize_context(input_data.get("context", ""))
    format_vars = {
        "context": context,
        "nodes": node_list,
        "transitions": transitions,
        "jit_operators": jit_operators,
        "jit_analysis_operators": jit_analysis_operators,
        "example_edges": example_edges(),
        "market_feed_config": load_asset_json_str("market_feed.json"),
        "market_analysis_config": load_asset_json_str("market_analysis.json"),
        "analysis_schema": load_asset_json_str("analysis_decision_schema.json"),
        "trading_policy": load_asset_json_str("trading_policy.json"),
        "trade_schema": load_asset_json_str("trade_decision_schema.json"),
    }
    prompt = templates[args.template].format_map(format_vars)

    import urllib.request
    import urllib.error

    if GROUP_CHAT_URL and "/gg/" in GROUP_CHAT_URL:
        content = _call_group_chat(prompt, GROUP_CHAT_URL)
    else:
        content = _call_completions(prompt)

    print(content)
    sys.exit(0)


def _call_group_chat(prompt: str, group_chat_url: str) -> str:
    """Send prompt to an existing ChatGPT group chat room and return the reply text.

    Uses POST /actions/group-chat with {"target_url": ..., "message": ...}.
    The browser-router drives the browser tab at the /gg/ URL and returns the
    assistant reply in the "content" field.
    """
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "target_url": group_chat_url,
        "message": prompt,
    }).encode()

    req = urllib.request.Request(
        GROUP_CHAT_ACTION_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode())
            if not body.get("ok"):
                sys.stderr.write(f"group chat action returned ok=false: {body}\n")
                sys.exit(1)
            content = body.get("content", "")
            if not content:
                sys.stderr.write(f"group chat action returned empty content: {body}\n")
                sys.exit(1)
            return content
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace").strip()
        detail = f": {error_body}" if error_body else ""
        sys.stderr.write(f"group chat HTTP error {exc.code}: {exc.reason}{detail}\n")
        sys.exit(1)
    except urllib.error.URLError as exc:
        sys.stderr.write(f"browser-router connection failed (group chat): {exc.reason}\n")
        sys.exit(127)
    except (KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"unexpected group chat response shape: {exc}\n")
        sys.exit(1)


def _call_completions(prompt: str) -> str:
    """Send prompt via /v1/chat/completions and return the reply text."""
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
            return body["choices"][0]["message"]["content"]
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
