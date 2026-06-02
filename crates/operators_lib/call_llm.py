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
ASSET_DIR             = pathlib.Path(__file__).resolve().parent.parent.parent / "assets"


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
        "\n\nDEVELOPMENT CYCLE — propose when context shows failures, stagnant learning, or stub nodes:\n"
        "  State::Learn -> State::Introspect  (confidence 0.3-0.5) — trigger topology self-development\n"
        "  State::Learn -> State::PatternPlan (confidence 0.2-0.4) — evolve CKA patterns\n"
        "Include at most one development edge per plan.\n"
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
    "topology_mutate": (
        "You are a neuro-symbolic topology architect embedded in a quantale-matrix agent loop.\n"
        "Your output is applied directly to the live topology graph — be conservative and precise.\n\n"
        "Diagnostic report from State::Introspect:\n{context}\n\n"
        "Current topology nodes:\n{nodes}\n\n"
        "Current topology transitions:\n{transitions}\n\n"
        "Agent goal metrics:\n{goal_metrics}\n\n"
        "Propose topology mutations that address failures and declining edges in the diagnostic report.\n"
        "Prioritise: fixing high-failure nodes, adding missing paths, adjusting declining edge weights.\n"
        "New operator nodes must have an operator_contract with executable=\'true\' as stub or an existing .py file.\n"
        "Do NOT delete: Control::Halt, Control::Retry, Control::Repair, Control::GateExecution.\n"
        "If no mutation is justified, emit an empty topology_ops array.\n\n"
        "Node types: State, Control, Event, Execution, Analysis\n"
        "Edge weights: default_weight, confidence, cost, safety — floats in [0.0,1.0]; cost=1-confidence baseline.\n\n"
        "Output ONLY a JSON object — no prose, no markdown fences:\n"
        '{{"topology_ops":[{{"op":"create_node","node":{{"name":"Namespace::Name","type":"State"}}}},{{"op":"create_edge","from":"A","to":"B","default_weight":0.5,"confidence":0.5,"cost":0.5,"safety":0.9}},{{"op":"update_edge","from":"A","to":"B","patch":{{"confidence":0.8,"cost":0.2}}}},{{"op":"delete_edge","from":"A","to":"B"}}],"operator_contracts":[{{"node_name":"Namespace::Name","executable":"true","static_args":[],"input_mapping":{{"stdin_mode":"json"}},"effects":{{"reads":[],"writes":[],"locks":[]}}}}],"reason":"one sentence"}}\n'
    ),
    "operator_write": (
        "You are a neuro-symbolic operator developer. Write a complete Python operator file.\n\n"
        "System context (goal, architecture, existing operators, stubs):\n{system_context}\n\n"
        "Recent context (topology mutations just applied):\n{context}\n\n"
        "Agent goal metrics:\n{goal_metrics}\n\n"
        "Choose the most important stub from system_context and implement it.\n"
        "Write a complete Python operator. File goes to crates/operators_lib/<filename>.\n\n"
        "RULES:\n"
        "- Unwrap context envelopes from stdin: payload may arrive as {{\"context\":\"<json>\"}}\n"
        "- Print one JSON object to stdout. Exit 0 success, 1 error.\n"
        "- Use only Python stdlib. No third-party imports.\n"
        "- Asset paths: pathlib.Path(__file__).resolve().parent.parent.parent / \"assets\"\n"
        "- Module docstring: node name, what it does, input/output shape.\n"
        "- Under 150 lines. Follow existing operator patterns from system_context.\n\n"
        "Also emit operator_contract_ops to upgrade executable=\'true\' to python3.\n"
        "Output ONLY a JSON object — no prose, no markdown fences:\n"
        '{{"filename":"snake_case.py","node_name":"Namespace::Name","source":"#!/usr/bin/env python3\\n...","operator_contract_ops":[{{"op":"update","node_name":"Namespace::Name","patch":{{"executable":"python3","static_args":["crates/operators_lib/snake_case.py"],"input_mapping":{{"stdin_mode":"json"}}}}}}]}}\n'
    ),
    "pattern_mutate": (
        "You are a neuro-symbolic CKA pattern architect in a quantale-matrix agent loop.\n"
        "Patterns define seq/par/choice/star execution structures for the batch scheduler.\n\n"
        "Prior execution context:\n{context}\n\n"
        "Current topology nodes:\n{nodes}\n\n"
        "Agent goal metrics:\n{goal_metrics}\n\n"
        "Propose pattern mutations to improve batch execution. Only reference existing nodes.\n"
        "par requires effect-independent nodes. Do not delete identity_skip_marker or blocked_marker.\n\n"
        "Expr grammar: string | {{\"seq\":[...]}} | {{\"choice\":[...]}} | {{\"par\":[...]}} | {{\"star\":{{\"body\":{{}},\"max_unroll\":3}}}}\n\n"
        "Output ONLY a JSON object — no prose, no markdown fences:\n"
        '{{"pattern_ops":[{{"op":"create","pattern":{{"name":"n","expr":{{"seq":["A","B"]}},"confidence":0.9,"cost":1.0,"safety":0.9}}}},{{"op":"update","name":"x","patch":{{"confidence":0.8}}}},{{"op":"delete","name":"y"}}],"reason":"one sentence"}}\n'
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


def load_latest_market_snapshot() -> str:
    """Return a compact price table from the most recent market_feed.jsonl entry.

    Falls back to "(market feed unavailable)" if the file is absent or empty.
    The snapshot is injected into the trade template so the LLM sees real prices
    even when the execution context carries no fresh market data.
    """
    state_log = pathlib.Path("state") / "market_feed.jsonl"
    if not state_log.exists():
        state_log = ASSET_DIR.parent / "state" / "market_feed.jsonl"
    last_line = ""
    try:
        with state_log.open() as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
    except OSError:
        return "(market feed unavailable)"
    if not last_line:
        return "(market feed unavailable — no entries)"
    try:
        obj = json.loads(last_line)
        feed = obj.get("market_feed", {})
        observed_at = feed.get("observed_at", "unknown time")
        symbols = feed.get("symbols", [])
        if not symbols:
            return f"(market feed empty at {observed_at})"
        lines = [f"Latest prices as of {observed_at}:"]
        for entry in symbols:
            sym = entry.get("symbol", "?")
            price = entry.get("price")
            open_ = entry.get("open")
            high = entry.get("high")
            low = entry.get("low")
            change_pct = ""
            if price is not None and open_ is not None and open_ != 0:
                pct = (price - open_) / open_ * 100
                change_pct = f"  change_from_open={pct:+.2f}%"
            lines.append(
                f"  {sym}: price={price}  open={open_}  high={high}  low={low}{change_pct}"
            )
        return "\n".join(lines)
    except (json.JSONDecodeError, AttributeError):
        return "(market feed parse error)"


def load_goal_metrics() -> str:
    """Compact runtime metrics: paper PnL trend and step failure rate."""
    metrics: dict = {}

    fills_path = pathlib.Path("state/paper_fills.jsonl")
    if not fills_path.exists():
        fills_path = ASSET_DIR.parent / "state" / "paper_fills.jsonl"
    fills = []
    try:
        for line in fills_path.read_text().splitlines()[-20:]:
            line = line.strip()
            if line:
                fills.append(json.loads(line))
    except OSError:
        pass
    if fills:
        net = sum(f.get("notional", 0) * (1 if f.get("side") == "sell" else -1) for f in fills)
        metrics["recent_fills"] = len(fills)
        metrics["net_notional_last_fills"] = round(net, 2)

    tlog_path = pathlib.Path("state/quantale.tlog")
    if not tlog_path.exists():
        tlog_path = ASSET_DIR.parent / "state" / "quantale.tlog"
    steps = []
    try:
        for line in tlog_path.read_text().splitlines()[-400:]:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("kind") == "AgentStep":
                    steps.append(r["payload"])
            except (json.JSONDecodeError, KeyError):
                pass
    except OSError:
        pass
    if steps:
        failures = sum(1 for s in steps if s.get("exit_code", 0) != 0)
        metrics["step_failure_rate"] = round(failures / len(steps), 3)
        metrics["steps_sampled"] = len(steps)

    return json.dumps(metrics, separators=(",", ":")) if metrics else "(metrics unavailable)"


def load_system_context() -> str:
    """Return a compact system context: goal, architecture excerpt, existing operator list."""
    parts = []

    goal_path = ASSET_DIR.parent / "GOAL.md"
    try:
        parts.append("=== GOAL ===\n" + goal_path.read_text().strip()[:800])
    except OSError:
        pass

    arch_path = ASSET_DIR.parent / "ARCHITECTURE.md"
    try:
        lines = arch_path.read_text().splitlines()[:40]
        parts.append("=== ARCHITECTURE (excerpt) ===\n" + "\n".join(lines))
    except OSError:
        pass

    ops_lib = ASSET_DIR.parent / "crates" / "operators_lib"
    op_lines = []
    try:
        for p in sorted(ops_lib.glob("*.py")):
            try:
                first_lines = p.read_text().splitlines()
                doc = next(
                    (l.strip().strip('"\'') for l in first_lines[1:10]
                     if l.strip() and not l.strip().startswith("#")),
                    ""
                )
                op_lines.append(f"  {p.name}: {doc[:80]}")
            except OSError:
                pass
    except OSError:
        pass
    if op_lines:
        parts.append("=== EXISTING OPERATORS ===\n" + "\n".join(op_lines))

    try:
        ops = json.loads(OPERATORS_PATH.read_text()).get("operators", [])
        stub_lines = [
            f"  {op['node_name']}: {op.get('description','(no description)')}  effects={json.dumps(op.get('effects',{}),separators=(',',':'))}"
            for op in ops if op.get("executable") == "true"
        ]
        if stub_lines:
            parts.append("=== STUB OPERATORS (need implementation) ===\n" + "\n".join(stub_lines[:15]))
    except Exception:
        pass

    return "\n\n".join(parts) if parts else "(system context unavailable)"


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
        "market_snapshot": load_latest_market_snapshot(),
        "goal_metrics": load_goal_metrics(),
        "system_context": load_system_context(),
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
