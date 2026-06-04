use super::*;

pub(super) fn maybe_hard_reset_after_blocks(
    consecutive_blocks: &mut usize,
    world: &mut TensorQuantaleWorld,
    accumulated_edges: &[quantale_semiring_v2::TensorEdge],
    current_payload: &mut Value,
    hard_reset_sleep: std::time::Duration,
    projection_bias: ProjectionBias,
    runtime_context: &RuntimeContext,
) {
    if *consecutive_blocks == 0 {
        return;
    }

    console::warn(
        "runtime",
        "hard_reset",
        &[("consecutive_blocks", consecutive_blocks.to_string())],
    );
    if let Err(error) = world.reset() {
        console::warn(
            "runtime",
            "hard_reset_world_reset_failed",
            &[("error", error.to_string())],
        );
    }
    if let Err(error) = world.embed_tensor_edges(accumulated_edges) {
        console::warn(
            "runtime",
            "hard_reset_embed_failed",
            &[("error", error.to_string())],
        );
    }
    if let Err(error) = world.close() {
        console::warn(
            "runtime",
            "hard_reset_close_failed",
            &[("error", error.to_string())],
        );
    }
    *current_payload = runtime_context.reset_payload();
    *consecutive_blocks = 0;
    std::thread::sleep(hard_reset_sleep);

    if let Ok(post_reset) = world.project(projection_bias) {
        if post_reset.blocked != 0 {
            console::warn(
                "runtime",
                "hard_reset_frontier_invalid",
                &[("first_hop", post_reset.first_hop.to_string())],
            );
        }
    }
}
