//! Build-time math compiler for generated `jit_cuda` operators.
//!
//! `assets/math.source.json` is the canonical formula layer.  It is compiled by
//! `topology build-overlay` into ordinary operator contracts with `jit_body` and
//! slot effects.  The runtime then executes those contracts through the existing
//! `FusionDispatch -> JitChain -> JitCache -> CUDA` path.

use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledMathOperator {
    pub node_name: String,
    pub contract: Value,
}

/// Compile all formulas in `math.source.json` into generated operator contracts.
///
/// Expected shape:
///
/// ```json
/// {
///   "formulas": [
///     {
///       "name": "return_1",
///       "node": "Analysis::Return1",
///       "inputs": ["market.price", "market.open"],
///       "output": "analysis.return",
///       "expr": {"div": [{"sub": ["market.price", "market.open"]}, {"add": ["market.open", 1e-8]}]}
///     }
///   ]
/// }
/// ```
pub fn compile_math_source(
    math: &Value,
    topology_source: &Value,
) -> Result<Vec<CompiledMathOperator>, String> {
    let formulas = math
        .get("formulas")
        .and_then(Value::as_array)
        .ok_or_else(|| "math.source.json must contain array field 'formulas'".to_string())?;

    let slots = collect_tensor_slots(topology_source)?;
    let nodes = collect_node_names(topology_source)?;
    let mut seen_nodes = BTreeSet::new();
    let mut compiled = Vec::with_capacity(formulas.len());

    for (idx, formula) in formulas.iter().enumerate() {
        let ctx = format!("math.formulas[{idx}]");
        let name = required_str(formula, "name", &ctx)?;
        let node_name = required_str(formula, "node", &ctx)?;
        if !nodes.contains(node_name) {
            return Err(format!(
                "{ctx} '{name}': node '{node_name}' does not exist in topology.source.json"
            ));
        }
        if !seen_nodes.insert(node_name.to_string()) {
            return Err(format!(
                "{ctx} '{name}': duplicate formula for node '{node_name}'"
            ));
        }

        let inputs = required_str_array(formula, "inputs", &ctx)?;
        if inputs.is_empty() {
            return Err(format!("{ctx} '{name}': inputs must not be empty"));
        }
        if inputs.len() > 3 {
            return Err(format!(
                "{ctx} '{name}': jit_cuda operators currently support 1..=3 input slots, got {}",
                inputs.len()
            ));
        }
        let mut input_seen = BTreeSet::new();
        for slot in &inputs {
            if !input_seen.insert(slot.clone()) {
                return Err(format!("{ctx} '{name}': duplicate input slot '{slot}'"));
            }
            if !slots.contains(slot) {
                return Err(format!(
                    "{ctx} '{name}': input slot '{slot}' is missing or is not a tensor f32[] slot"
                ));
            }
        }

        let output = required_str(formula, "output", &ctx)?;
        if !slots.contains(output) {
            return Err(format!(
                "{ctx} '{name}': output slot '{output}' is missing or is not a tensor f32[] slot"
            ));
        }
        if inputs.iter().any(|slot| slot == output) {
            return Err(format!(
                "{ctx} '{name}': output slot '{output}' must not also be an input slot"
            ));
        }

        let expr = formula
            .get("expr")
            .ok_or_else(|| format!("{ctx} '{name}': missing expr"))?;
        let input_map = inputs
            .iter()
            .enumerate()
            .map(|(i, slot)| (slot.as_str(), format!("in{i}[i]")))
            .collect::<BTreeMap<_, _>>();
        let body_expr = compile_expr(expr, &input_map, &format!("{ctx} '{name}' expr"))?;
        let jit_body = format!("out[i] = {body_expr};");

        let description = formula
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("Generated CUDA math formula '{name}'"));

        let contract = json!({
            "node_name": node_name,
            "executable": "jit_cuda",
            "jit_body": jit_body,
            "description": description,
            "effects": {
                "reads": inputs,
                "writes": [output],
                "locks": []
            }
        });

        compiled.push(CompiledMathOperator {
            node_name: node_name.to_string(),
            contract,
        });
    }

    Ok(compiled)
}

/// Replace matching operator contracts by node name, or append if absent.
pub fn apply_compiled_math_operators(
    operator_contracts: &mut Vec<Value>,
    compiled: Vec<CompiledMathOperator>,
) -> Result<(), String> {
    for item in compiled {
        let mut replaced = false;
        for contract in operator_contracts.iter_mut() {
            if contract.get("node_name").and_then(Value::as_str) == Some(item.node_name.as_str()) {
                *contract = item.contract.clone();
                replaced = true;
                break;
            }
        }
        if !replaced {
            operator_contracts.push(item.contract);
        }
    }
    Ok(())
}

fn required_str<'a>(value: &'a Value, field: &str, ctx: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("{ctx}: field '{field}' must be a non-empty string"))
}

fn required_str_array(value: &Value, field: &str, ctx: &str) -> Result<Vec<String>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{ctx}: field '{field}' must be an array of strings"))?
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            item.as_str()
                .filter(|s| !s.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| format!("{ctx}: field '{field}[{idx}]' must be a non-empty string"))
        })
        .collect()
}

fn collect_node_names(source: &Value) -> Result<BTreeSet<String>, String> {
    let nodes = source
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| "topology.source.json must contain array field 'nodes'".to_string())?;
    Ok(nodes
        .iter()
        .filter_map(|node| node.get("name").and_then(Value::as_str).map(str::to_string))
        .collect())
}

fn collect_tensor_slots(source: &Value) -> Result<BTreeSet<String>, String> {
    let slots = source
        .get("slots")
        .and_then(Value::as_object)
        .ok_or_else(|| "topology.source.json must contain object field 'slots'".to_string())?;
    Ok(slots
        .iter()
        .filter_map(|(name, slot)| {
            let ty = slot.get("type").and_then(Value::as_str);
            let kind = slot.get("kind").and_then(Value::as_str);
            (ty == Some("f32[]") && kind == Some("tensor")).then(|| name.clone())
        })
        .collect())
}

fn compile_expr(
    expr: &Value,
    inputs: &BTreeMap<&str, String>,
    ctx: &str,
) -> Result<String, String> {
    match expr {
        Value::Number(n) => compile_number(n, ctx),
        Value::String(slot) => inputs
            .get(slot.as_str())
            .cloned()
            .ok_or_else(|| format!("{ctx}: unknown input slot '{slot}'")),
        Value::Object(obj) => compile_op(obj, inputs, ctx),
        _ => Err(format!(
            "{ctx}: expression must be number, input slot string, or operator object"
        )),
    }
}

fn compile_number(n: &serde_json::Number, ctx: &str) -> Result<String, String> {
    let value = n
        .as_f64()
        .ok_or_else(|| format!("{ctx}: number is not representable as f64"))?;
    if !value.is_finite() {
        return Err(format!("{ctx}: numeric literal must be finite"));
    }
    // Use a suffix so integer JSON literals still become valid float expressions.
    Ok(format_float_literal(value))
}

fn compile_op(
    obj: &Map<String, Value>,
    inputs: &BTreeMap<&str, String>,
    ctx: &str,
) -> Result<String, String> {
    if obj.len() != 1 {
        return Err(format!(
            "{ctx}: operator object must contain exactly one key"
        ));
    }
    let (op, args) = obj.iter().next().unwrap();
    match op.as_str() {
        "add" => compile_nary(args, inputs, ctx, op, 2, |items| {
            format!("({})", items.join(" + "))
        }),
        "mul" => compile_nary(args, inputs, ctx, op, 2, |items| {
            format!("({})", items.join(" * "))
        }),
        "sub" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("({} - {})", items[0], items[1])
        }),
        "div" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("({} / {})", items[0], items[1])
        }),
        "min" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("fminf({}, {})", items[0], items[1])
        }),
        "max" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("fmaxf({}, {})", items[0], items[1])
        }),
        "pow" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("powf({}, {})", items[0], items[1])
        }),
        "atan2" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("atan2f({}, {})", items[0], items[1])
        }),
        "abs" => compile_unary(args, inputs, ctx, op, |x| format!("fabsf({x})")),
        "neg" => compile_unary(args, inputs, ctx, op, |x| format!("(-{x})")),
        "sqrt" => compile_unary(args, inputs, ctx, op, |x| {
            format!("sqrtf(fmaxf(0.0f, {x}))")
        }),
        "exp" => compile_unary(args, inputs, ctx, op, |x| format!("expf({x})")),
        "log" => compile_unary(args, inputs, ctx, op, |x| {
            format!("logf(fmaxf(1.0e-20f, {x}))")
        }),
        "tanh" => compile_unary(args, inputs, ctx, op, |x| format!("tanhf({x})")),
        "sin" => compile_unary(args, inputs, ctx, op, |x| format!("sinf({x})")),
        "cos" => compile_unary(args, inputs, ctx, op, |x| format!("cosf({x})")),
        "sigmoid" => compile_unary(args, inputs, ctx, op, |x| {
            format!("(1.0f / (1.0f + expf(-({x}))))")
        }),
        "clamp" => compile_fixed(args, inputs, ctx, op, 3, |items| {
            format!("fminf(fmaxf({}, {}), {})", items[0], items[1], items[2])
        }),
        "where" => compile_fixed(args, inputs, ctx, op, 3, |items| {
            format!("(({}) ? ({}) : ({}))", items[0], items[1], items[2])
        }),
        "gt" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("({} > {})", items[0], items[1])
        }),
        "gte" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("({} >= {})", items[0], items[1])
        }),
        "lt" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("({} < {})", items[0], items[1])
        }),
        "lte" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("({} <= {})", items[0], items[1])
        }),
        "eq" => compile_fixed(args, inputs, ctx, op, 2, |items| {
            format!("({} == {})", items[0], items[1])
        }),
        "and" => compile_nary(args, inputs, ctx, op, 2, |items| {
            format!("({})", items.join(" && "))
        }),
        "or" => compile_nary(args, inputs, ctx, op, 2, |items| {
            format!("({})", items.join(" || "))
        }),
        "not" => compile_unary(args, inputs, ctx, op, |x| format!("(!({x}))")),
        other => Err(format!(
            "{ctx}: unsupported math operator '{other}' (allowed: add/sub/mul/div/min/max/pow/atan2/abs/neg/sqrt/exp/log/tanh/sin/cos/sigmoid/clamp/where/comparisons/and/or/not)"
        )),
    }
}

fn compile_unary<F>(
    args: &Value,
    inputs: &BTreeMap<&str, String>,
    ctx: &str,
    op: &str,
    emit: F,
) -> Result<String, String>
where
    F: FnOnce(String) -> String,
{
    let items = compile_arg_array(args, inputs, ctx, op)?;
    if items.len() != 1 {
        return Err(format!(
            "{ctx}: operator '{op}' expects exactly 1 argument, got {}",
            items.len()
        ));
    }
    Ok(emit(items.into_iter().next().unwrap()))
}

fn compile_fixed<F>(
    args: &Value,
    inputs: &BTreeMap<&str, String>,
    ctx: &str,
    op: &str,
    expected: usize,
    emit: F,
) -> Result<String, String>
where
    F: FnOnce(Vec<String>) -> String,
{
    let items = compile_arg_array(args, inputs, ctx, op)?;
    if items.len() != expected {
        return Err(format!(
            "{ctx}: operator '{op}' expects exactly {expected} arguments, got {}",
            items.len()
        ));
    }
    Ok(emit(items))
}

fn compile_nary<F>(
    args: &Value,
    inputs: &BTreeMap<&str, String>,
    ctx: &str,
    op: &str,
    min_args: usize,
    emit: F,
) -> Result<String, String>
where
    F: FnOnce(Vec<String>) -> String,
{
    let items = compile_arg_array(args, inputs, ctx, op)?;
    if items.len() < min_args {
        return Err(format!(
            "{ctx}: operator '{op}' expects at least {min_args} arguments, got {}",
            items.len()
        ));
    }
    Ok(emit(items))
}

fn compile_arg_array(
    args: &Value,
    inputs: &BTreeMap<&str, String>,
    ctx: &str,
    op: &str,
) -> Result<Vec<String>, String> {
    let arr = args
        .as_array()
        .ok_or_else(|| format!("{ctx}: operator '{op}' expects an array of arguments"))?;
    arr.iter()
        .enumerate()
        .map(|(idx, arg)| compile_expr(arg, inputs, &format!("{ctx}.{op}[{idx}]")))
        .collect()
}

fn format_float_literal(value: f64) -> String {
    if value == 0.0 {
        return "0.0f".to_string();
    }
    let abs = value.abs();
    let raw = if !(1.0e-4..1.0e7).contains(&abs) {
        format!("{value:.9e}")
    } else if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        let mut s = format!("{value:.9}");
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        s
    };
    format!("{raw}f")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn topology() -> Value {
        json!({
            "slots": {
                "a": {"type":"f32[]", "kind":"tensor"},
                "b": {"type":"f32[]", "kind":"tensor"},
                "c": {"type":"f32[]", "kind":"tensor"},
                "json.slot": {"type":"json", "kind":"state"}
            },
            "nodes": [
                {"name":"Math::Add", "kind":"kernel"},
                {"name":"Math::Score", "kind":"kernel"}
            ]
        })
    }

    #[test]
    fn compiles_formula_to_jit_contract() {
        let math = json!({
            "formulas": [{
                "name": "add",
                "node": "Math::Add",
                "inputs": ["a", "b"],
                "output": "c",
                "expr": {"add": ["a", "b"]}
            }]
        });
        let compiled = compile_math_source(&math, &topology()).unwrap();
        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].node_name, "Math::Add");
        assert_eq!(compiled[0].contract["executable"], "jit_cuda");
        assert_eq!(
            compiled[0].contract["jit_body"],
            "out[i] = (in0[i] + in1[i]);"
        );
        assert_eq!(compiled[0].contract["effects"]["reads"], json!(["a", "b"]));
        assert_eq!(compiled[0].contract["effects"]["writes"], json!(["c"]));
    }

    #[test]
    fn compiles_nested_formula() {
        let math = json!({
            "formulas": [{
                "name": "score",
                "node": "Math::Score",
                "inputs": ["a", "b"],
                "output": "c",
                "expr": {"div": ["a", {"add": [1.0, {"abs": ["b"]}]}]}
            }]
        });
        let compiled = compile_math_source(&math, &topology()).unwrap();
        assert_eq!(
            compiled[0].contract["jit_body"],
            "out[i] = (in0[i] / (1.0f + fabsf(in1[i])));"
        );
    }

    #[test]
    fn rejects_unknown_slot() {
        let math = json!({
            "formulas": [{
                "name": "bad",
                "node": "Math::Add",
                "inputs": ["a", "missing"],
                "output": "c",
                "expr": {"add": ["a", "missing"]}
            }]
        });
        let err = compile_math_source(&math, &topology()).unwrap_err();
        assert!(err.contains("missing"), "{err}");
    }

    #[test]
    fn replaces_existing_operator_contract() {
        let mut operators = vec![json!({
            "node_name": "Math::Add",
            "executable": "true",
            "effects": {"reads": [], "writes": [], "locks": []}
        })];
        let compiled = vec![CompiledMathOperator {
            node_name: "Math::Add".into(),
            contract: json!({"node_name":"Math::Add", "executable":"jit_cuda"}),
        }];
        apply_compiled_math_operators(&mut operators, compiled).unwrap();
        assert_eq!(operators.len(), 1);
        assert_eq!(operators[0]["executable"], "jit_cuda");
    }
}
