use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JitChain {
    pub operators: Vec<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub internals: Vec<String>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JitChainMetadata {
    pub chain_len: i32,
    pub input_count: i32,
    pub output_count: i32,
    pub estimated_savings: f32,
    pub target_node_id: i32,
}

unsafe impl cudarc::driver::DeviceRepr for JitChainMetadata {}

pub fn detect_jit_chains(
    operator_names: &[String],
    registry: &HashMap<String, Value>,
) -> Result<Vec<JitChain>, String> {
    let mut chains = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for name in operator_names {
        if !is_jit_cuda(name, registry) {
            flush_chain(&mut current, &mut chains, registry)?;
            continue;
        }

        if current.is_empty() || chain_links_to_operator(&current, name, registry)? {
            current.push(name.clone());
        } else {
            flush_chain(&mut current, &mut chains, registry)?;
            current.push(name.clone());
        }
    }

    flush_chain(&mut current, &mut chains, registry)?;
    Ok(chains)
}

pub fn chain_for_single_operator(
    operator_name: &str,
    registry: &HashMap<String, Value>,
) -> Result<JitChain, String> {
    build_chain(&[operator_name.to_string()], registry)
}

pub fn chain_metadata(chain: &JitChain, target_node_id: i32) -> JitChainMetadata {
    let launches_saved = chain.operators.len().saturating_sub(1) as f32;
    let internal_round_trips = chain.internals.len() as f32;
    JitChainMetadata {
        chain_len: chain.operators.len() as i32,
        input_count: chain.inputs.len() as i32,
        output_count: chain.outputs.len() as i32,
        estimated_savings: launches_saved + 2.0 * internal_round_trips,
        target_node_id,
    }
}

fn is_jit_cuda(name: &str, registry: &HashMap<String, Value>) -> bool {
    registry
        .get(name)
        .and_then(|op| op.get("executable"))
        .and_then(Value::as_str)
        == Some("jit_cuda")
}

fn chain_links_to_operator(
    current: &[String],
    next: &str,
    registry: &HashMap<String, Value>,
) -> Result<bool, String> {
    let mut writes = BTreeSet::new();
    for name in current {
        writes.extend(effect_slots(name, registry, "writes")?);
    }
    let reads = effect_slots(next, registry, "reads")?;
    Ok(reads.iter().any(|slot| writes.contains(slot)))
}

fn flush_chain(
    current: &mut Vec<String>,
    chains: &mut Vec<JitChain>,
    registry: &HashMap<String, Value>,
) -> Result<(), String> {
    if !current.is_empty() {
        chains.push(build_chain(current, registry)?);
        current.clear();
    }
    Ok(())
}

fn build_chain(names: &[String], registry: &HashMap<String, Value>) -> Result<JitChain, String> {
    let mut produced = BTreeSet::new();
    let mut consumed = BTreeSet::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for name in names {
        if !is_jit_cuda(name, registry) {
            return Err(format!("operator '{name}' is not executable=jit_cuda"));
        }
        for slot in effect_slots(name, registry, "reads")? {
            if !produced.contains(&slot) && !inputs.contains(&slot) {
                inputs.push(slot.clone());
            }
            consumed.insert(slot);
        }
        for slot in effect_slots(name, registry, "writes")? {
            produced.insert(slot);
        }
    }

    let internals: Vec<String> = produced.intersection(&consumed).cloned().collect();
    for slot in produced {
        if !internals.contains(&slot) {
            outputs.push(slot);
        }
    }

    Ok(JitChain {
        operators: names.to_vec(),
        inputs,
        outputs,
        internals,
    })
}

pub(crate) fn effect_slots(
    name: &str,
    registry: &HashMap<String, Value>,
    field: &str,
) -> Result<Vec<String>, String> {
    let op = registry
        .get(name)
        .ok_or_else(|| format!("operator '{name}' missing from registry"))?;
    let arr = op
        .get("effects")
        .and_then(|effects| effects.get(field))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("operator '{name}' effects.{field} must be an array"))?;
    arr.iter()
        .map(|slot| {
            slot.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("operator '{name}' effects.{field} must contain strings"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn registry() -> HashMap<String, Value> {
        HashMap::from([
            (
                "A".to_string(),
                json!({"executable":"jit_cuda","effects":{"reads":["x","y"],"writes":["z"],"locks":[]}}),
            ),
            (
                "B".to_string(),
                json!({"executable":"jit_cuda","effects":{"reads":["z","s"],"writes":["o"],"locks":[]}}),
            ),
            (
                "C".to_string(),
                json!({"executable":"true","effects":{"reads":["o"],"writes":["done"],"locks":[]}}),
            ),
        ])
    }

    #[test]
    fn groups_consecutive_dependent_jit_operators() {
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let chains = detect_jit_chains(&names, &registry()).unwrap();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].inputs, vec!["x", "y", "s"]);
        assert_eq!(chains[0].outputs, vec!["o"]);
        assert_eq!(chains[0].internals, vec!["z"]);
    }
}
