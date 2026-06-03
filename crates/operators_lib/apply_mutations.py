#!/usr/bin/env python3
"""Review or apply pending records from state/mutation_queue.jsonl.

This is the explicit escape hatch for queued repo mutations. It replays each
pending record through the original operator with QUANTALE_MUTATION_MODE=apply
only when --apply is provided.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
import os
import pathlib
import subprocess
import sys

import mutation_policy
import pattern_mutate
import topology_mutate

SCRIPT_BY_KIND = {
    "operator_write": "write_operator.py",
    "topology_patch": "topology_mutate.py",
    "pattern_patch": "pattern_mutate.py",
}
POLICY_PATH = mutation_policy.PROJECT_ROOT / "assets" / "mutation_review_policy.json"
GOVERNANCE_PATH = mutation_policy.PROJECT_ROOT / "assets" / "governance_policy.json"
OPERATORS_JSON = mutation_policy.PROJECT_ROOT / "assets" / "operators.json"


def _load_records(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _write_records(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def _load_policy() -> dict:
    try:
        return json.loads(POLICY_PATH.read_text())
    except Exception:
        return {}


def _load_governance_policy() -> dict:
    try:
        return json.loads(GOVERNANCE_PATH.read_text())
    except Exception:
        return {}


def _json_text(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _diff(path: str, before: str, after: str, max_chars: int) -> dict:
    text = "".join(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    ))
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n...[truncated]\n"
    return {"path": path, "diff": text, "truncated": truncated}


def _edge_key(edge: dict) -> tuple[str, str]:
    return (edge.get("from", ""), edge.get("to", ""))


def _governance_findings(record: dict) -> list[dict]:
    policy = _load_governance_policy()
    limits = policy.get("mutation_limits", {})
    protected_nodes = set(policy.get("protected_nodes", []))
    protected_edges = {
        (edge.get("from", ""), edge.get("to", ""))
        for edge in policy.get("protected_edges", [])
    }
    findings = []
    kind = record.get("kind", "")
    payload = record.get("payload", {})

    if kind == "operator_write":
        filename = payload.get("filename", "")
        source = payload.get("source", "")
        source_bytes = len(source.encode())
        max_source_bytes = int(limits.get("max_operator_source_bytes", 20000))
        if source_bytes > max_source_bytes:
            findings.append({
                "severity": "block",
                "reason": f"operator source is {source_bytes} bytes; max is {max_source_bytes}",
            })
        if filename in set(policy.get("blocked_operator_filenames", [])):
            findings.append({
                "severity": "block",
                "reason": f"operator filename is protected: {filename}",
            })
        try:
            tree = ast.parse(source)
            blocked_imports = set(policy.get("operator_source", {}).get("blocked_imports", []))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if alias.name in blocked_imports or root in blocked_imports:
                            findings.append({
                                "severity": "block",
                                "reason": f"operator source imports blocked module: {alias.name}",
                            })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    root = module.split(".")[0]
                    if module in blocked_imports or root in blocked_imports:
                        findings.append({
                            "severity": "block",
                            "reason": f"operator source imports blocked module: {module}",
                        })
        except SyntaxError as exc:
            findings.append({"severity": "block", "reason": f"operator syntax error: {exc}"})

    if kind == "topology_patch":
        topology_ops = payload.get("topology_ops", [])
        max_topology_ops = int(limits.get("max_topology_ops", 8))
        if len(topology_ops) > max_topology_ops:
            findings.append({
                "severity": "block",
                "reason": f"topology_ops count is {len(topology_ops)}; max is {max_topology_ops}",
            })
        for op in topology_ops:
            op_kind = op.get("op", "")
            node_name = op.get("name") or op.get("node", {}).get("name", "")
            if op_kind in {"delete_node", "replace_node"} and node_name in protected_nodes:
                findings.append({
                    "severity": "block",
                    "reason": f"{op_kind} targets protected node {node_name}",
                })
            edge = _edge_key(op)
            if op_kind in {"delete_edge", "replace_edge"} and edge in protected_edges:
                findings.append({
                    "severity": "block",
                    "reason": f"{op_kind} targets protected edge {edge[0]} -> {edge[1]}",
                })
        contracts = payload.get("operator_contracts", [])
        contract_ops = payload.get("operator_contract_ops", [])
        max_contracts = int(limits.get("max_operator_contracts", 4))
        max_contract_ops = int(limits.get("max_operator_contract_ops", 6))
        if len(contracts) > max_contracts:
            findings.append({
                "severity": "block",
                "reason": f"operator_contracts count is {len(contracts)}; max is {max_contracts}",
            })
        if len(contract_ops) > max_contract_ops:
            findings.append({
                "severity": "block",
                "reason": f"operator_contract_ops count is {len(contract_ops)}; max is {max_contract_ops}",
            })

    if kind == "pattern_patch":
        pattern_ops = payload.get("pattern_ops", [])
        max_pattern_ops = int(limits.get("max_pattern_ops", 8))
        if len(pattern_ops) > max_pattern_ops:
            findings.append({
                "severity": "block",
                "reason": f"pattern_ops count is {len(pattern_ops)}; max is {max_pattern_ops}",
            })

    return findings


def _apply_operator_contract_ops(operators: dict, ops: list) -> tuple[list, str | None]:
    updated = []
    for op in ops:
        kind = op.get("op")
        if kind == "update":
            name = op.get("node_name", "")
            patch = op.get("patch", {})
            idx = next((i for i, item in enumerate(operators["operators"])
                        if item.get("node_name") == name), None)
            if idx is None:
                return updated, f"update operator_contract: not found: {name}"
            operators["operators"][idx].update({k: v for k, v in patch.items() if k != "node_name"})
            updated.append({"op": "update", "node_name": name})
        elif kind == "replace":
            contract = op.get("contract", {})
            name = contract.get("node_name", "")
            idx = next((i for i, item in enumerate(operators["operators"])
                        if item.get("node_name") == name), None)
            if idx is None:
                return updated, f"replace operator_contract: not found: {name}"
            operators["operators"][idx] = contract
            updated.append({"op": "replace", "node_name": name})
        else:
            return updated, f"unknown operator_contract_op: {kind!r}"
    return updated, None


def _preview_operator_write(record: dict, max_chars: int) -> dict:
    payload = record.get("payload", {})
    filename = payload.get("filename", "")
    source = payload.get("source", "")
    target = mutation_policy.PROJECT_ROOT / "crates" / "operators_lib" / filename
    before = target.read_text() if target.exists() else ""
    diffs = [_diff(str(target.relative_to(mutation_policy.PROJECT_ROOT)), before, source, max_chars)]

    ops = payload.get("operator_contract_ops", [])
    if ops:
        operators = json.loads(OPERATORS_JSON.read_text())
        proposed = copy.deepcopy(operators)
        _, error = _apply_operator_contract_ops(proposed, ops)
        if error:
            return {"error": error, "diffs": diffs}
        diffs.append(_diff(
            str(OPERATORS_JSON.relative_to(mutation_policy.PROJECT_ROOT)),
            _json_text(operators),
            _json_text(proposed),
            max_chars,
        ))

    return {"diffs": diffs}


def _preview_topology_patch(record: dict, max_chars: int) -> dict:
    payload = record.get("payload", {})
    topology = json.loads(topology_mutate.TOPOLOGY_PATH.read_text())
    proposed_topology = copy.deepcopy(topology)
    _, error = topology_mutate._apply_ops(proposed_topology, payload.get("topology_ops", []))
    if error:
        return {"error": error, "diffs": []}

    diffs = [_diff(
        str(topology_mutate.TOPOLOGY_PATH.relative_to(mutation_policy.PROJECT_ROOT)),
        _json_text(topology),
        _json_text(proposed_topology),
        max_chars,
    )]

    contracts = payload.get("operator_contracts", [])
    contract_ops = payload.get("operator_contract_ops", [])
    if contracts or contract_ops:
        operators = json.loads(topology_mutate.OPERATORS_PATH.read_text())
        proposed_operators = copy.deepcopy(operators)
        _, error = topology_mutate._apply_contracts(proposed_operators, contracts)
        if error:
            return {"error": error, "diffs": diffs}
        _, error = topology_mutate._apply_contract_ops(proposed_operators, contract_ops)
        if error:
            return {"error": error, "diffs": diffs}
        diffs.append(_diff(
            str(topology_mutate.OPERATORS_PATH.relative_to(mutation_policy.PROJECT_ROOT)),
            _json_text(operators),
            _json_text(proposed_operators),
            max_chars,
        ))

    return {"diffs": diffs}


def _preview_pattern_patch(record: dict, max_chars: int) -> dict:
    payload = record.get("payload", {})
    patterns = json.loads(pattern_mutate.PATTERNS_PATH.read_text())
    proposed = copy.deepcopy(patterns)
    _, error = pattern_mutate._apply_ops(proposed, payload.get("pattern_ops", []))
    if error:
        return {"error": error, "diffs": []}
    return {"diffs": [_diff(
        str(pattern_mutate.PATTERNS_PATH.relative_to(mutation_policy.PROJECT_ROOT)),
        _json_text(patterns),
        _json_text(proposed),
        max_chars,
    )]}


def _preview_record(record: dict, policy: dict) -> dict:
    max_chars = int(policy.get("preview", {}).get("max_diff_chars", 20000))
    kind = record.get("kind", "")
    if kind == "operator_write":
        preview = _preview_operator_write(record, max_chars)
    elif kind == "topology_patch":
        preview = _preview_topology_patch(record, max_chars)
    elif kind == "pattern_patch":
        preview = _preview_pattern_patch(record, max_chars)
    else:
        preview = {"error": f"unknown mutation kind: {kind!r}", "diffs": []}
    if policy.get("preview", {}).get("include_payload", False):
        preview["payload"] = record.get("payload", {})
    preview["governance"] = _governance_findings(record)
    return preview


def _apply_record(record: dict) -> dict:
    findings = _governance_findings(record)
    blockers = [finding for finding in findings if finding.get("severity") == "block"]
    if blockers:
        return {
            "exit_code": 2,
            "stdout": "",
            "stderr": "governance policy blocked mutation",
            "governance": blockers,
        }

    script_name = SCRIPT_BY_KIND.get(record.get("kind", ""))
    if not script_name:
        return {"exit_code": 1, "stderr": f"unknown mutation kind: {record.get('kind')!r}"}

    script_path = pathlib.Path(__file__).resolve().parent / script_name
    env = os.environ.copy()
    env["QUANTALE_MUTATION_MODE"] = "apply"
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        input=json.dumps(record.get("payload", {})),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _list_records(records: list[dict], mutation_id: str = "", with_preview: bool = False) -> dict:
    policy = _load_policy()
    pending = []
    for record in records:
        if mutation_id and record.get("id") != mutation_id:
            continue
        if record.get("status") != "pending":
            continue
        item = {
            "id": record.get("id", ""),
            "ts": record.get("ts", ""),
            "kind": record.get("kind", ""),
            "source_node": record.get("source_node", ""),
            "effects": record.get("effects", []),
            "target_paths": record.get("target_paths", []),
            "summary": record.get("summary", {}),
        }
        if with_preview:
            item["preview"] = _preview_record(record, policy)
        pending.append(item)
    return {"pending": pending, "pending_count": len(pending)}


def _reject_records(records: list[dict], mutation_id: str) -> list[dict]:
    rejected = []
    for record in records:
        if record.get("status") != "pending":
            continue
        if mutation_id and record.get("id") != mutation_id:
            continue
        record["status"] = "rejected"
        rejected.append({"id": record.get("id"), "status": "rejected"})
    return rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", dest="mutation_id", default="")
    parser.add_argument("--queue", default=str(mutation_policy.DEFAULT_QUEUE_PATH))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reject", action="store_true")
    args = parser.parse_args()

    queue_path = pathlib.Path(args.queue)
    records = _load_records(queue_path)
    policy = _load_policy()

    if args.reject:
        if not args.mutation_id and not policy.get("reject", {}).get("allow_reject_all", False):
            print(json.dumps({"mutation_queue": {"error": "reject all is disabled by policy"}}))
            sys.exit(1)
        rejected = _reject_records(records, args.mutation_id)
        _write_records(queue_path, records)
        print(json.dumps({"mutation_queue": {"records": rejected, "queue_path": str(queue_path)}}))
        return

    if args.list or args.preview or not args.apply:
        listed = _list_records(records, args.mutation_id, with_preview=args.preview)
        listed["queue_path"] = str(queue_path)
        print(json.dumps({"mutation_queue": listed}))
        return

    if not args.mutation_id and not policy.get("apply", {}).get("allow_apply_all", True):
        print(json.dumps({"apply_mutations": {"error": "apply all is disabled by policy"}}))
        sys.exit(1)

    applied = []
    for record in records:
        if record.get("status") != "pending":
            continue
        if args.mutation_id and record.get("id") != args.mutation_id:
            continue
        result = _apply_record(record)
        record["apply_result"] = result
        record["status"] = "applied" if result["exit_code"] == 0 else "failed"
        applied.append({"id": record.get("id"), "status": record["status"]})

    _write_records(queue_path, records)
    print(json.dumps({"apply_mutations": {"records": applied, "queue_path": str(queue_path)}}))


if __name__ == "__main__":
    main()
