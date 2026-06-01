//! Concurrent token-value exploration engine.
//!
//! This is the host reference model for the exploration layer. It is deliberately
//! data-driven: strategy configuration comes from `assets/exploration.json`,
//! legal node IDs come from topology, effects come from `operators.json`, and
//! receipt priors are updated from execution receipts.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use cudarc::driver::DeviceRepr;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::OperatorRegistry;
use crate::error::CudaError;
use crate::node::{Node, START_NODE, node_name};
use crate::rule_delta::ProcessReceipt;
use crate::tensor::{
    COST_INFINITY, LAYER_CONFIDENCE, LAYER_COST, LAYER_SAFETY, ProjectionBias, tensor_idx,
};
use crate::topology::GraphTopology;

pub const DEFAULT_EXPLORATION_JSON: &str = include_str!("../assets/exploration.json");

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplorationConfig {
    pub beam_width: usize,
    pub max_depth: usize,
    pub max_batches: usize,
    pub novelty_weight: f32,
    pub receipt_weight: f32,
    pub entropy_penalty: f32,
    pub repeat_penalty: f32,
    pub max_terminal_visits: usize,
    pub max_first_hop_visits: usize,
    pub strategies: Vec<ExplorationStrategy>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ExplorationConfigFile {
    engine: ExplorationEngineConfig,
    strategies: Vec<ExplorationStrategy>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ExplorationEngineConfig {
    beam_width: usize,
    max_depth: usize,
    max_batches: usize,
    novelty_weight: f32,
    receipt_weight: f32,
    entropy_penalty: f32,
    #[serde(default = "default_repeat_penalty")]
    repeat_penalty: f32,
    #[serde(default = "default_max_visits")]
    max_terminal_visits: usize,
    #[serde(default = "default_max_visits")]
    max_first_hop_visits: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplorationStrategy {
    pub name: String,
    pub start: String,
    pub bias: ProjectionBias,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExplorationToken {
    pub strategy_id: i32,
    pub node: i32,
    pub depth: i32,
    pub confidence: f32,
    pub cost: f32,
    pub safety: f32,
    pub novelty: f32,
    pub receipt_prior: f32,
    pub entropy: f32,
    pub parent: i32,
}

unsafe impl DeviceRepr for ExplorationToken {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExplorationCandidate {
    pub token_id: i32,
    pub first_hop: i32,
    pub terminal_node: i32,
    pub value: f32,
}

unsafe impl DeviceRepr for ExplorationCandidate {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplorationBatch {
    pub step: i32,
    pub candidates: Vec<ExplorationCandidate>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplorationCommitRecord {
    pub strategy: String,
    pub depth: i32,
    pub candidate_count: usize,
    pub winner: ExplorationCandidate,
    pub path: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExplorationDecision {
    CommitExploration(ExplorationCandidate),
    UseCkaBatch,
    SingleFrontier,
    Blocked,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExplorationEngine {
    config: ExplorationConfig,
    adjacency: Vec<Vec<i32>>,
    operator_registry: OperatorRegistry,
    receipt_priors: HashMap<i32, f32>,
    terminal_visits: HashMap<i32, i32>,
    first_hop_visits: HashMap<i32, i32>,
    tokens: Vec<ExplorationToken>,
    selected: Vec<ExplorationCandidate>,
}

impl ExplorationConfig {
    pub fn from_json_str(input: &str) -> Result<Self, CudaError> {
        let file: ExplorationConfigFile = serde_json::from_str(input).map_err(|error| {
            CudaError::invalid_input(format!("parse exploration config: {error}"))
        })?;
        let config = Self {
            beam_width: file.engine.beam_width,
            max_depth: file.engine.max_depth,
            max_batches: file.engine.max_batches,
            novelty_weight: file.engine.novelty_weight,
            receipt_weight: file.engine.receipt_weight,
            entropy_penalty: file.engine.entropy_penalty,
            repeat_penalty: file.engine.repeat_penalty,
            max_terminal_visits: file.engine.max_terminal_visits,
            max_first_hop_visits: file.engine.max_first_hop_visits,
            strategies: file.strategies,
        };
        config.validate_basic()?;
        Ok(config)
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CudaError> {
        let input = fs::read_to_string(path.as_ref()).map_err(|error| {
            CudaError::invalid_input(format!(
                "read exploration config '{}': {error}",
                path.as_ref().display()
            ))
        })?;
        Self::from_json_str(&input)
    }

    pub fn default_asset() -> Result<Self, CudaError> {
        Self::from_json_str(DEFAULT_EXPLORATION_JSON)
    }

    pub fn validate_against_topology(&self, topology: &GraphTopology) -> Result<(), CudaError> {
        let compiled = topology.compile()?;
        for strategy in &self.strategies {
            if compiled.registry.id_of(&strategy.start).is_none() {
                return Err(CudaError::invalid_input(format!(
                    "unknown exploration strategy node '{}'",
                    strategy.start
                )));
            }
        }
        Ok(())
    }

    fn validate_basic(&self) -> Result<(), CudaError> {
        if self.beam_width == 0 {
            return Err(CudaError::invalid_input(
                "exploration beam_width must be > 0",
            ));
        }
        if self.max_depth == 0 {
            return Err(CudaError::invalid_input(
                "exploration max_depth must be > 0",
            ));
        }
        if self.max_batches == 0 {
            return Err(CudaError::invalid_input(
                "exploration max_batches must be > 0",
            ));
        }
        if self.repeat_penalty < 0.0 || !self.repeat_penalty.is_finite() {
            return Err(CudaError::invalid_input(
                "exploration repeat_penalty must be finite and >= 0",
            ));
        }
        if self.max_terminal_visits == 0 || self.max_first_hop_visits == 0 {
            return Err(CudaError::invalid_input(
                "exploration max visit thresholds must be > 0",
            ));
        }
        if self.strategies.is_empty() {
            return Err(CudaError::invalid_input(
                "exploration requires at least one strategy",
            ));
        }
        for strategy in &self.strategies {
            if strategy.name.trim().is_empty() || strategy.start.trim().is_empty() {
                return Err(CudaError::invalid_input(
                    "exploration strategy name/start must be non-empty",
                ));
            }
            if !strategy.bias.confidence.is_finite()
                || !strategy.bias.cost.is_finite()
                || !strategy.bias.safety.is_finite()
            {
                return Err(CudaError::invalid_input(
                    "exploration strategy bias values must be finite",
                ));
            }
        }
        Ok(())
    }
}

impl ExplorationEngine {
    pub fn new(
        config: ExplorationConfig,
        topology: &GraphTopology,
        operator_registry: OperatorRegistry,
    ) -> Result<Self, CudaError> {
        config.validate_against_topology(topology)?;
        let compiled = topology.compile()?;
        let mut adjacency = vec![Vec::new(); compiled.node_count];
        for edge in &compiled.tensor_edges {
            let src = usize::try_from(edge.src)
                .map_err(|_| CudaError::invalid_input("negative exploration edge src"))?;
            if src < adjacency.len() && Node::decode(edge.dst).is_some() {
                adjacency[src].push(edge.dst);
            }
        }
        Ok(Self {
            config,
            adjacency,
            operator_registry,
            receipt_priors: HashMap::new(),
            terminal_visits: HashMap::new(),
            first_hop_visits: HashMap::new(),
            tokens: Vec::new(),
            selected: Vec::new(),
        })
    }

    pub fn from_default_assets(operator_registry: OperatorRegistry) -> Result<Self, CudaError> {
        let config = ExplorationConfig::default_asset()?;
        let topology = GraphTopology::default_asset()?;
        Self::new(config, &topology, operator_registry)
    }

    pub fn config(&self) -> &ExplorationConfig {
        &self.config
    }

    pub fn tokens(&self) -> &[ExplorationToken] {
        &self.tokens
    }

    pub fn selected(&self) -> &[ExplorationCandidate] {
        &self.selected
    }

    pub fn strategy_nodes(&self) -> Result<Vec<i32>, CudaError> {
        self.config
            .strategies
            .iter()
            .map(|strategy| node_id_from_name(&strategy.start))
            .collect()
    }

    pub fn strategy_biases(&self) -> Vec<ProjectionBias> {
        self.config
            .strategies
            .iter()
            .map(|strategy| strategy.bias)
            .collect()
    }

    pub fn receipt_prior_vector(&self) -> Vec<f32> {
        let mut priors = vec![0.0; crate::node::NODE_COUNT];
        for (node, prior) in &self.receipt_priors {
            if let Ok(index) = usize::try_from(*node) {
                if index < priors.len() {
                    priors[index] = *prior;
                }
            }
        }
        priors
    }
    pub fn terminal_visit_vector(&self) -> Vec<i32> {
        visit_vector(&self.terminal_visits)
    }

    pub fn first_hop_visit_vector(&self) -> Vec<i32> {
        visit_vector(&self.first_hop_visits)
    }

    pub fn terminal_visit_count(&self, node: i32) -> i32 {
        self.terminal_visits.get(&node).copied().unwrap_or(0)
    }

    pub fn first_hop_visit_count(&self, node: i32) -> i32 {
        self.first_hop_visits.get(&node).copied().unwrap_or(0)
    }

    pub fn mark_candidate_committed(&mut self, candidate: &ExplorationCandidate) {
        *self
            .terminal_visits
            .entry(candidate.terminal_node)
            .or_insert(0) += 1;
        *self
            .first_hop_visits
            .entry(candidate.first_hop)
            .or_insert(0) += 1;
    }

    pub fn candidate_allowed_by_repeat_policy(&self, candidate: &ExplorationCandidate) -> bool {
        self.terminal_visit_count(candidate.terminal_node) < self.config.max_terminal_visits as i32
            && self.first_hop_visit_count(candidate.first_hop)
                < self.config.max_first_hop_visits as i32
    }

    pub fn best_commit_candidate(&self) -> Option<ExplorationCandidate> {
        self.selected.iter().copied().find(|candidate| {
            self.candidate_allowed_by_repeat_policy(candidate)
                && self.validate_candidate_effect(candidate).is_ok()
        })
    }

    pub fn load_gpu_state(
        &mut self,
        tokens: Vec<ExplorationToken>,
        selected: Vec<ExplorationCandidate>,
    ) {
        self.tokens = tokens;
        self.selected = selected;
    }

    pub fn seed_tokens(&mut self, tensor: &[f32]) -> Result<&[ExplorationToken], CudaError> {
        self.tokens.clear();
        let start = START_NODE.encode();
        for (strategy_id, strategy) in self.config.strategies.iter().enumerate() {
            let node = node_id_from_name(&strategy.start)?;
            let confidence = tensor_value(tensor, LAYER_CONFIDENCE, start, node, 0.0);
            let safety = tensor_value(tensor, LAYER_SAFETY, start, node, 0.0);
            if confidence <= 0.0 && safety <= 0.0 {
                continue;
            }
            self.tokens.push(ExplorationToken {
                strategy_id: strategy_id as i32,
                node,
                depth: 0,
                confidence: confidence * strategy.bias.confidence,
                cost: tensor_cost(tensor, start, node) * strategy.bias.cost,
                safety: safety * strategy.bias.safety,
                novelty: novelty_for_node(node),
                receipt_prior: self.receipt_prior_for(node),
                entropy: entropy_for_node(node),
                parent: -1,
            });
        }
        Ok(&self.tokens)
    }

    pub fn host_expand_exploration(
        &mut self,
        tensor: &[f32],
    ) -> Result<Vec<ExplorationCandidate>, CudaError> {
        validate_tensor_len(tensor)?;
        if self.tokens.is_empty() {
            self.seed_tokens(tensor)?;
        }
        let mut frontier: Vec<i32> = (0..self.tokens.len() as i32).collect();
        let mut candidates = Vec::new();
        for depth in 0..self.config.max_depth {
            if depth > 0 {
                candidates.extend(
                    frontier
                        .iter()
                        .map(|token_id| self.candidate_for_token(*token_id)),
                );
            }
            let mut next_frontier = Vec::new();
            for &token_id in &frontier {
                let token = self.tokens[token_id as usize];
                let Some(children) = usize::try_from(token.node)
                    .ok()
                    .and_then(|idx| self.adjacency.get(idx))
                else {
                    continue;
                };
                for &child in children {
                    let confidence = tensor_value(tensor, LAYER_CONFIDENCE, token.node, child, 0.0);
                    let cost = tensor_cost(tensor, token.node, child);
                    let safety = tensor_value(tensor, LAYER_SAFETY, token.node, child, 0.0);
                    if confidence <= 0.0 && safety <= 0.0 {
                        continue;
                    }
                    self.tokens.push(ExplorationToken {
                        strategy_id: token.strategy_id,
                        node: child,
                        depth: token.depth + 1,
                        confidence: token.confidence + confidence,
                        cost: token.cost + cost,
                        safety: token.safety + safety,
                        novelty: token.novelty + novelty_for_node(child),
                        receipt_prior: token.receipt_prior + self.receipt_prior_for(child),
                        entropy: token.entropy + entropy_for_node(child),
                        parent: token_id,
                    });
                    next_frontier.push((self.tokens.len() - 1) as i32);
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            next_frontier.sort_by(|a, b| {
                self.token_value(*b)
                    .partial_cmp(&self.token_value(*a))
                    .unwrap_or(Ordering::Equal)
            });
            if next_frontier.len() > self.config.beam_width {
                next_frontier.truncate(self.config.beam_width);
            }
            frontier = next_frontier;
        }
        candidates.extend(
            frontier
                .iter()
                .map(|token_id| self.candidate_for_token(*token_id)),
        );
        let selected = self.host_select_topk(candidates);
        self.selected = selected.clone();
        Ok(selected)
    }

    pub fn host_select_topk(
        &self,
        mut candidates: Vec<ExplorationCandidate>,
    ) -> Vec<ExplorationCandidate> {
        candidates.retain(|candidate| self.candidate_allowed_by_repeat_policy(candidate));
        candidates.sort_by(|a, b| {
            self.repeat_adjusted_value(*b)
                .partial_cmp(&self.repeat_adjusted_value(*a))
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.terminal_node.cmp(&b.terminal_node))
        });
        candidates.dedup_by_key(|candidate| candidate.terminal_node);
        for candidate in &mut candidates {
            candidate.value = self.repeat_adjusted_value(*candidate);
        }
        if candidates.len() > self.config.beam_width {
            candidates.truncate(self.config.beam_width);
        }
        candidates
    }

    pub fn propose(
        &mut self,
        tensor: &[f32],
        cka_batch_ready: bool,
        single_frontier_ready: bool,
    ) -> Result<ExplorationDecision, CudaError> {
        self.host_expand_exploration(tensor)?;
        if let Some(candidate) = self.best_commit_candidate() {
            return Ok(ExplorationDecision::CommitExploration(candidate));
        }
        if cka_batch_ready {
            Ok(ExplorationDecision::UseCkaBatch)
        } else if single_frontier_ready {
            Ok(ExplorationDecision::SingleFrontier)
        } else {
            Ok(ExplorationDecision::Blocked)
        }
    }

    pub fn validate_candidate_effect(
        &self,
        candidate: &ExplorationCandidate,
    ) -> Result<(), CudaError> {
        let terminal = node_name(candidate.terminal_node);
        let Some(operator) = self.operator_registry.get(&terminal) else {
            return Ok(());
        };
        let locks = operator
            .get("effects")
            .and_then(|effects| effects.get("locks"))
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let has_exclusive_lock = locks
            .iter()
            .filter_map(Value::as_str)
            .any(|lock| matches!(lock, "workspace" | "executor" | "memory" | "learning"));
        if has_exclusive_lock
            && !matches!(
                terminal.as_str(),
                "Control::GateExecution" | "State::Validate"
            )
        {
            return Err(CudaError::invalid_input(format!(
                "exploration candidate '{}' requires exclusive unsafe lock",
                terminal
            )));
        }
        Ok(())
    }

    pub fn reconstruct_exploration_path(&self, candidate: ExplorationCandidate) -> Vec<Node> {
        let mut out = Vec::new();
        let mut cursor = candidate.token_id;
        while cursor >= 0 {
            let Some(token) = self.tokens.get(cursor as usize) else {
                break;
            };
            if let Some(node) = Node::decode(token.node) {
                out.push(node);
            }
            cursor = token.parent;
        }
        out.reverse();
        out
    }

    pub fn commit_record(&self, candidate: ExplorationCandidate) -> ExplorationCommitRecord {
        let strategy = self
            .tokens
            .get(candidate.token_id as usize)
            .and_then(|token| self.config.strategies.get(token.strategy_id as usize))
            .map(|strategy| strategy.name.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let depth = self
            .tokens
            .get(candidate.token_id as usize)
            .map(|token| token.depth)
            .unwrap_or_default();
        let path = self
            .reconstruct_exploration_path(candidate)
            .into_iter()
            .map(|node| node.name().to_string())
            .collect();
        ExplorationCommitRecord {
            strategy,
            depth,
            candidate_count: self.selected.len(),
            winner: candidate,
            path,
        }
    }

    pub fn update_receipt_prior(&mut self, node: i32, receipt: &ProcessReceipt) {
        let current = self.receipt_prior_for(node);
        let observation = match receipt.exit_code {
            0 => 1.0,
            124 => -0.5,
            _ => -0.25,
        };
        let updated = (current * 0.8) + (observation * 0.2);
        self.receipt_priors.insert(node, updated.clamp(-1.0, 1.0));
    }

    pub fn receipt_prior_for(&self, node: i32) -> f32 {
        self.receipt_priors.get(&node).copied().unwrap_or(0.0)
    }

    fn repeat_adjusted_value(&self, candidate: ExplorationCandidate) -> f32 {
        let visits = self.terminal_visit_count(candidate.terminal_node)
            + self.first_hop_visit_count(candidate.first_hop);
        candidate.value - self.config.repeat_penalty * visits as f32
    }

    fn candidate_for_token(&self, token_id: i32) -> ExplorationCandidate {
        let token = self.tokens[token_id as usize];
        ExplorationCandidate {
            token_id,
            first_hop: self.first_hop(token_id),
            terminal_node: token.node,
            value: self.token_value(token_id),
        }
    }

    fn first_hop(&self, token_id: i32) -> i32 {
        let mut current = token_id;
        let mut parent = self.tokens[current as usize].parent;
        while parent >= 0 {
            let grandparent = self.tokens[parent as usize].parent;
            if grandparent < 0 {
                return self.tokens[current as usize].node;
            }
            current = parent;
            parent = grandparent;
        }
        self.tokens[token_id as usize].node
    }

    fn token_value(&self, token_id: i32) -> f32 {
        let token = self.tokens[token_id as usize];
        let depth = (token.depth + 1) as f32;
        (token.confidence / depth) - (token.cost / depth)
            + (token.safety / depth)
            + self.config.novelty_weight * (token.novelty / depth)
            + self.config.receipt_weight * (token.receipt_prior / depth)
            - self.config.entropy_penalty * (token.entropy / depth)
    }
}

fn default_repeat_penalty() -> f32 {
    1.25
}

fn default_max_visits() -> usize {
    1
}

fn visit_vector(visits: &HashMap<i32, i32>) -> Vec<i32> {
    let mut out = vec![0; crate::node::NODE_COUNT];
    for (node, count) in visits {
        if let Ok(index) = usize::try_from(*node) {
            if index < out.len() {
                out[index] = *count;
            }
        }
    }
    out
}

fn validate_tensor_len(tensor: &[f32]) -> Result<(), CudaError> {
    if tensor.len() < crate::tensor::TENSOR_LEN {
        return Err(CudaError::invalid_input(format!(
            "exploration tensor length {} < {}",
            tensor.len(),
            crate::tensor::TENSOR_LEN
        )));
    }
    Ok(())
}

fn node_id_from_name(name: &str) -> Result<i32, CudaError> {
    let topology = GraphTopology::default_asset()?.compile()?;
    topology
        .registry
        .id_of(name)
        .map(|id| id as i32)
        .ok_or_else(|| CudaError::invalid_input(format!("unknown exploration node '{name}'")))
}

fn tensor_value(tensor: &[f32], layer: i32, src: i32, dst: i32, fallback: f32) -> f32 {
    let value = tensor[tensor_idx(layer, src, dst)];
    if value.is_finite() { value } else { fallback }
}

fn tensor_cost(tensor: &[f32], src: i32, dst: i32) -> f32 {
    let value = tensor[tensor_idx(LAYER_COST, src, dst)];
    if value.is_finite() && value < COST_INFINITY {
        value
    } else {
        COST_INFINITY
    }
}

fn novelty_for_node(node: i32) -> f32 {
    // Stable deterministic novelty proxy until CUDA frontier heat is wired in.
    ((node.rem_euclid(7) + 1) as f32) / 10.0
}

fn entropy_for_node(node: i32) -> f32 {
    ((node.rem_euclid(5) + 1) as f32) / 10.0
}
