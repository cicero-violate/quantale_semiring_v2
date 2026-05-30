use quantale_semiring_v2::{CudaWorld, format_quantale_value, node_name};

fn main() {
    let mut world = match CudaWorld::from_edges(&quantale_semiring_v2::full_transition_edges()) {
        Ok(world) => world,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    for _ in 0..4 {
        let report = match world.step() {
            Ok(report) => report,
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        };
        let decision = match world.project_decision_path() {
            Ok(decision) => decision,
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        };

        println!(
            "step={} best=({}->{}) value={} events={} Goal->Execute={} Goal->Learn={} projection=({}->{}) witness_first_hop={} dvalue={} selected_action={:?} halted={} blocked={}",
            report.step,
            node_name(report.best_src),
            node_name(report.best_dst),
            format_quantale_value(report.best_value),
            report.event_count,
            format_quantale_value(report.goal_to_execute),
            format_quantale_value(report.goal_to_learn),
            node_name(decision.selected_src),
            node_name(decision.selected_dst),
            node_name(decision.first_hop),
            format_quantale_value(decision.selected_value),
            decision.selected_action(),
            decision.halted,
            decision.blocked,
        );
    }
}
