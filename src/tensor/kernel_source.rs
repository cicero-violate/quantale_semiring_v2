use super::CUDA_TENSOR_NODE_COUNT_DEFINE;

const CUDA_KERNEL_FRAGMENTS: &[&str] = &[
    include_str!("../../cuda/quantale/00_prelude.cuh"),
    include_str!("../../cuda/quantale/01_state_and_rings.cuh"),
    include_str!("../../cuda/quantale/02_control_flow_abi.cuh"),
    include_str!("../../cuda/quantale/03_tensor_core.cuh"),
    include_str!("../../cuda/quantale/04_exploration.cuh"),
    include_str!("../../cuda/quantale/05_jit_and_receipts.cuh"),
    include_str!("../../cuda/quantale/06_scheduler.cuh"),
    include_str!("../../cuda/quantale/07_failure_policy.cuh"),
    include_str!("../../cuda/quantale/08_learning.cuh"),
    include_str!("../../cuda/quantale/09_trace_replay.cuh"),
    include_str!("../../cuda/quantale/10_device_float_ring.cuh"),
    include_str!("../../cuda/quantale/11_hot_regions_dispatch.cuh"),
    include_str!("../../cuda/quantale/12_par_group.cuh"),
];

const FUSION_HF_GENERATED_FUNCTIONS_MARKER: &str = "// @@FUSION_HF_GENERATED_FUNCTIONS@@";
const FUSION_HF_GENERATED_GPU_DISPATCH_CASES_MARKER: &str =
    "// @@FUSION_HF_GENERATED_GPU_DISPATCH_CASES@@";
const FUSION_HF_GENERATED_PAR_CASES_MARKER: &str = "// @@FUSION_HF_GENERATED_PAR_CASES@@";
const GENERATED_FUSION_HF_STUBS_PATH: &str = "assets/fusion_hf.stubs.cu";

pub(super) fn assemble_kernel_source() -> String {
    let generated_functions = std::fs::read_to_string(GENERATED_FUSION_HF_STUBS_PATH)
        .unwrap_or_else(|_| "// No generated fusion H_f fragments available.\n".to_string());
    assemble_kernel_source_with_generated(&generated_functions)
}

pub(super) fn assemble_kernel_source_with_generated(generated_functions: &str) -> String {
    let mut source = format!(
        "{}{}",
        CUDA_TENSOR_NODE_COUNT_DEFINE,
        CUDA_KERNEL_FRAGMENTS.join("\n")
    );
    let generated_gpu_cases =
        generated_fusion_hf_switch_cases(generated_functions, "element_count");
    let generated_par_cases = generated_fusion_hf_switch_cases(generated_functions, "elem_count");

    source = source.replace(FUSION_HF_GENERATED_FUNCTIONS_MARKER, generated_functions);
    source = source.replace(
        FUSION_HF_GENERATED_GPU_DISPATCH_CASES_MARKER,
        &generated_gpu_cases,
    );
    source = source.replace(FUSION_HF_GENERATED_PAR_CASES_MARKER, &generated_par_cases);
    source
}

fn generated_fusion_hf_switch_cases(generated_functions: &str, element_count_name: &str) -> String {
    generated_functions
        .lines()
        .filter_map(|line| parse_generated_fusion_hf_case(line, element_count_name))
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_generated_fusion_hf_case(line: &str, element_count_name: &str) -> Option<String> {
    let line = line.trim();
    let rest = line.strip_prefix("// hf_case:")?.trim();
    let mut parts = rest.split_whitespace();
    let region_id = parts.next()?;
    let symbol = parts.next()?;
    Some(format!(
        "        case {region_id}: {symbol}(slot_ptrs, {element_count_name}, &r); break;"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_switch_cases_are_parsed_from_stub_metadata() {
        let gpu_cases = generated_fusion_hf_switch_cases(
            "// hf_case: 8 region_fusion_stub_fixture_add_fixture_scale\n",
            "element_count",
        );
        let par_cases = generated_fusion_hf_switch_cases(
            "// hf_case: 8 region_fusion_stub_fixture_add_fixture_scale\n",
            "elem_count",
        );
        assert!(gpu_cases.contains(
            "case 8: region_fusion_stub_fixture_add_fixture_scale(slot_ptrs, element_count, &r); break;"
        ));
        assert!(par_cases.contains(
            "case 8: region_fusion_stub_fixture_add_fixture_scale(slot_ptrs, elem_count, &r); break;"
        ));
    }

    #[test]
    fn kernel_source_assembler_replaces_generated_markers() {
        let source = assemble_kernel_source();
        assert!(!source.contains(FUSION_HF_GENERATED_FUNCTIONS_MARKER));
        assert!(!source.contains(FUSION_HF_GENERATED_GPU_DISPATCH_CASES_MARKER));
        assert!(!source.contains(FUSION_HF_GENERATED_PAR_CASES_MARKER));
    }

    #[test]
    fn cuda_fragment_source_keeps_generated_markers_once() {
        let source = CUDA_KERNEL_FRAGMENTS.join("\n");
        assert_eq!(
            source.matches(FUSION_HF_GENERATED_FUNCTIONS_MARKER).count(),
            1
        );
        assert_eq!(
            source
                .matches(FUSION_HF_GENERATED_GPU_DISPATCH_CASES_MARKER)
                .count(),
            1
        );
        assert_eq!(
            source.matches(FUSION_HF_GENERATED_PAR_CASES_MARKER).count(),
            1
        );
    }

    #[test]
    fn cuda_fragment_source_exports_expected_kernel_inventory() {
        let source = CUDA_KERNEL_FRAGMENTS.join("\n");
        let expected = [
            "orchestration_state_init",
            "orchestration_state_snapshot",
            "star_counters_init",
            "device_command_ring_push",
            "device_receipt_ext_ring_push",
            "device_receipt_ext_drain",
            "tensor_quantale_reset",
            "tensor_quantale_embed_edges",
            "tensor_quantale_closure",
            "tensor_quantale_project",
            "tensor_quantale_decay",
            "tensor_quantale_seed_exploration",
            "tensor_quantale_expand_tokens",
            "tensor_quantale_score_tokens",
            "tensor_quantale_select_topk_tokens",
            "tensor_quantale_commit_exploration",
            "jit_chain_score_embed",
            "tensor_quantale_drain_device_receipts",
            "tensor_quantale_push_device_receipt",
            "tensor_quantale_orchestrate_step",
            "check_effects_independent",
            "failure_policy_init",
            "failure_policy_classify_and_emit",
            "failure_policy_set_rollback_marker",
            "failure_policy_apply_rollback",
            "learned_delta_init",
            "learned_delta_fold_receipt",
            "learned_delta_apply",
            "receipt_prior_snapshot",
            "orch_event_trace_push",
            "orch_event_trace_drain",
            "orch_check_no_duplicate_receipts",
            "orch_check_frontier_valid",
            "orch_check_no_command_without_receipt",
            "orch_replay_snapshot",
            "orch_replay_restore",
            "device_ring_push",
            "device_ring_pop",
            "tensor_quantale_gpu_dispatch",
            "tensor_quantale_par_group_step",
        ];
        for kernel in expected {
            let needle = format!("extern \"C\" __global__ void {kernel}(");
            assert!(
                source.contains(&needle),
                "missing CUDA kernel symbol {kernel}"
            );
        }
        assert_eq!(
            source.matches("extern \"C\" __global__ void ").count(),
            expected.len()
        );
    }
}
