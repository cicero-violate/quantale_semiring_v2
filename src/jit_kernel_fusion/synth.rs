use std::collections::HashMap;

use serde_json::Value;

use super::chain::{JitChain, effect_slots};

pub const JIT_KERNEL_NAME: &str = "jit_fused";
pub const JIT_BATCH_KERNEL_NAME: &str = "jit_fused_batch";

pub fn synthesize_kernel(
    chain: &JitChain,
    registry: &HashMap<String, Value>,
) -> Result<String, String> {
    if chain.outputs.len() != 1 {
        return Err(format!(
            "JIT executor currently supports exactly one chain output, got {}",
            chain.outputs.len()
        ));
    }

    let mut source = String::new();
    source.push_str("extern \"C\" __global__ void ");
    source.push_str(JIT_KERNEL_NAME);
    source.push('(');
    for (idx, _) in chain.inputs.iter().enumerate() {
        source.push_str(&format!("const float* __restrict__ in{idx}, "));
    }
    source.push_str("float* __restrict__ out0, int n) {\n");
    source.push_str("    int i = blockIdx.x * blockDim.x + threadIdx.x;\n");
    source.push_str("    if (i >= n) return;\n");

    for op_name in &chain.operators {
        let op = registry
            .get(op_name)
            .ok_or_else(|| format!("operator '{op_name}' missing from registry"))?;
        let body = op
            .get("jit_body")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("operator '{op_name}' missing string jit_body"))?;
        let reads = effect_slots(op_name, registry, "reads")?;
        let writes = effect_slots(op_name, registry, "writes")?;
        if writes.len() != 1 {
            return Err(format!(
                "operator '{op_name}' must declare exactly one write slot for JIT synthesis"
            ));
        }
        let line = lower_body(body, &reads, &writes[0], chain)?;
        source.push_str("    ");
        source.push_str(&line);
        if !line.trim_end().ends_with(';') {
            source.push(';');
        }
        source.push('\n');
    }

    source.push_str("}\n");
    Ok(source)
}

pub fn synthesize_batch_kernel(
    chains: &[JitChain],
    registry: &HashMap<String, Value>,
) -> Result<String, String> {
    if chains.is_empty() {
        return Err("JIT batch synthesis requires at least one chain".to_string());
    }

    let mut source = String::new();
    source.push_str("extern \"C\" __global__ void ");
    source.push_str(JIT_BATCH_KERNEL_NAME);
    source.push('(');
    for (chain_idx, chain) in chains.iter().enumerate() {
        if chain.outputs.len() != 1 {
            return Err(format!(
                "JIT batch executor currently supports exactly one output per chain, got {} for chain {chain_idx}",
                chain.outputs.len()
            ));
        }
        for input_idx in 0..chain.inputs.len() {
            source.push_str(&format!(
                "const float* __restrict__ c{chain_idx}_in{input_idx}, "
            ));
        }
        source.push_str(&format!("float* __restrict__ c{chain_idx}_out0, "));
    }
    source.push_str("int n) {\n");
    source.push_str("    int i = blockIdx.x * blockDim.x + threadIdx.x;\n");
    source.push_str("    if (i >= n) return;\n");

    for (chain_idx, chain) in chains.iter().enumerate() {
        source.push_str(&format!("    // chain {chain_idx}\n"));
        for op_name in &chain.operators {
            let op = registry
                .get(op_name)
                .ok_or_else(|| format!("operator '{op_name}' missing from registry"))?;
            let body = op
                .get("jit_body")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("operator '{op_name}' missing string jit_body"))?;
            let reads = effect_slots(op_name, registry, "reads")?;
            let writes = effect_slots(op_name, registry, "writes")?;
            if writes.len() != 1 {
                return Err(format!(
                    "operator '{op_name}' must declare exactly one write slot for JIT batch synthesis"
                ));
            }
            let line = lower_body_with_names(
                body,
                &reads,
                &writes[0],
                chain,
                &format!("c{chain_idx}_in"),
                &format!("c{chain_idx}_out"),
                &format!("c{chain_idx}_reg"),
            )?;
            source.push_str("    ");
            source.push_str(&line);
            if !line.trim_end().ends_with(';') {
                source.push(';');
            }
            source.push('\n');
        }
    }

    source.push_str("}\n");
    Ok(source)
}

fn lower_body(
    body: &str,
    reads: &[String],
    write: &str,
    chain: &JitChain,
) -> Result<String, String> {
    lower_body_with_names(body, reads, write, chain, "in", "out", "reg")
}

fn lower_body_with_names(
    body: &str,
    reads: &[String],
    write: &str,
    chain: &JitChain,
    input_prefix: &str,
    output_prefix: &str,
    register_prefix: &str,
) -> Result<String, String> {
    let mut lowered = body.trim().trim_end_matches(';').to_string();
    for (idx, slot) in reads.iter().enumerate() {
        lowered = lowered.replace(
            &format!("in{idx}[i]"),
            &slot_expr_with_names(slot, chain, input_prefix, output_prefix, register_prefix)?,
        );
    }

    let write_expr =
        slot_expr_with_names(write, chain, input_prefix, output_prefix, register_prefix)?;
    if chain.internals.iter().any(|slot| slot == write) {
        let prefix = "out[i]";
        if let Some(expr) = lowered
            .strip_prefix(prefix)
            .and_then(|rest| rest.trim().strip_prefix('='))
        {
            return Ok(format!("float {write_expr} = {};", expr.trim()));
        }
    }

    lowered = lowered.replace("out[i]", &write_expr);
    Ok(format!("{lowered};"))
}

fn slot_expr_with_names(
    slot: &str,
    chain: &JitChain,
    input_prefix: &str,
    output_prefix: &str,
    register_prefix: &str,
) -> Result<String, String> {
    if let Some(idx) = chain.inputs.iter().position(|candidate| candidate == slot) {
        return Ok(format!("{input_prefix}{idx}[i]"));
    }
    if let Some(idx) = chain.outputs.iter().position(|candidate| candidate == slot) {
        return Ok(format!("{output_prefix}{idx}[i]"));
    }
    if chain.internals.iter().any(|candidate| candidate == slot) {
        return Ok(format!("{register_prefix}_{}", sanitize_ident(slot)));
    }
    Err(format!("slot '{slot}' is not part of JIT chain data flow"))
}

fn sanitize_ident(slot: &str) -> String {
    slot.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn emits_register_for_internal_slot() {
        let chain = JitChain {
            operators: vec!["add".to_string(), "scale".to_string()],
            inputs: vec!["a".to_string(), "b".to_string(), "scale".to_string()],
            outputs: vec!["out".to_string()],
            internals: vec!["tmp.add".to_string()],
        };
        let registry = HashMap::from([
            (
                "add".to_string(),
                json!({
                    "jit_body":"out[i] = in0[i] + in1[i];",
                    "effects":{"reads":["a","b"],"writes":["tmp.add"],"locks":[]}
                }),
            ),
            (
                "scale".to_string(),
                json!({
                    "jit_body":"out[i] = in0[i] * in1[i];",
                    "effects":{"reads":["tmp.add","scale"],"writes":["out"],"locks":[]}
                }),
            ),
        ]);
        let source = synthesize_kernel(&chain, &registry).unwrap();
        assert!(source.contains("float reg_tmp_add = in0[i] + in1[i];"));
        assert!(source.contains("out0[i] = reg_tmp_add * in2[i];"));
    }

    #[test]
    fn emits_batch_kernel_for_multiple_chains() {
        let chains = vec![
            JitChain {
                operators: vec!["add".to_string(), "scale".to_string()],
                inputs: vec!["a".to_string(), "b".to_string(), "scale".to_string()],
                outputs: vec!["out".to_string()],
                internals: vec!["tmp.add".to_string()],
            },
            JitChain {
                operators: vec!["ret".to_string()],
                inputs: vec!["price".to_string(), "open".to_string()],
                outputs: vec!["return".to_string()],
                internals: vec![],
            },
        ];
        let registry = HashMap::from([
            (
                "add".to_string(),
                json!({
                    "jit_body":"out[i] = in0[i] + in1[i];",
                    "effects":{"reads":["a","b"],"writes":["tmp.add"],"locks":[]}
                }),
            ),
            (
                "scale".to_string(),
                json!({
                    "jit_body":"out[i] = in0[i] * in1[i];",
                    "effects":{"reads":["tmp.add","scale"],"writes":["out"],"locks":[]}
                }),
            ),
            (
                "ret".to_string(),
                json!({
                    "jit_body":"out[i] = (in0[i] - in1[i]) / (in1[i] + 1e-8f);",
                    "effects":{"reads":["price","open"],"writes":["return"],"locks":[]}
                }),
            ),
        ]);

        let source = synthesize_batch_kernel(&chains, &registry).unwrap();
        assert!(source.contains("jit_fused_batch"));
        assert!(source.contains("const float* __restrict__ c0_in0"));
        assert!(source.contains("float* __restrict__ c0_out0"));
        assert!(source.contains("const float* __restrict__ c1_in1"));
        assert!(source.contains("float* __restrict__ c1_out0"));
        assert!(source.contains("float c0_reg_tmp_add = c0_in0[i] + c0_in1[i];"));
        assert!(source.contains("c0_out0[i] = c0_reg_tmp_add * c0_in2[i];"));
        assert!(source.contains("c1_out0[i] = (c1_in0[i] - c1_in1[i]) / (c1_in1[i] + 1e-8f);"));
    }
}
