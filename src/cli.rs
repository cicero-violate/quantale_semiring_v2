use super::*;

pub(super) enum CliCommand {
    ContinueRuntime,
    Exit(i32),
}

pub(super) fn handle(args: &[String]) -> CliCommand {
    if args.get(1).map(String::as_str) == Some("topology")
        && args.get(2).map(String::as_str) == Some("build-overlay")
    {
        if let Err(error) = build_overlay_assets(".") {
            console::error(
                "topology",
                "build_overlay_failed",
                &[("error", error.to_string())],
            );
            return CliCommand::Exit(1);
        }
        if let Err(error) = compile_and_emit_pattern_edges(".") {
            console::error(
                "topology",
                "pattern_compile_failed",
                &[("error", error.to_string())],
            );
            return CliCommand::Exit(1);
        }
        console::info(
            "topology",
            "overlay_written",
            &[
                ("topology", "assets/topology.generated.json".to_string()),
                ("operators", "assets/operators.generated.json".to_string()),
                ("patterns", "assets/patterns.compiled.json".to_string()),
            ],
        );
        return CliCommand::Exit(0);
    }

    if args.iter().any(|a| a == "--check-topology") {
        return check_topology();
    }

    CliCommand::ContinueRuntime
}

fn check_topology() -> CliCommand {
    let topology = match GraphTopology::default_asset() {
        Ok(t) => t,
        Err(error) => {
            console::error("topology", "parse_failed", &[("error", error.to_string())]);
            return CliCommand::Exit(1);
        }
    };
    let inv = TopologyInvariants::default_asset();
    let violations = check(&topology, &inv);
    let (warnings, fatal): (Vec<_>, Vec<_>) = violations
        .into_iter()
        .partition(|v| v.kind == ViolationKind::ConsumedBlockPoint);
    for v in &warnings {
        console::warn("topology", "violation", &[("detail", v.to_string())]);
    }
    if fatal.is_empty() {
        console::info(
            "topology",
            "ok",
            &[
                ("nodes", topology.nodes.len().to_string()),
                ("transitions", topology.transitions.len().to_string()),
                ("warnings", warnings.len().to_string()),
            ],
        );
        return CliCommand::Exit(0);
    }
    console::error(
        "topology",
        "violations",
        &[
            ("count", fatal.len().to_string()),
            ("detail", format_violations(&fatal)),
        ],
    );
    CliCommand::Exit(1)
}
