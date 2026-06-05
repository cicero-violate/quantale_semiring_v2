use super::*;

impl TensorQuantaleWorld {
    pub fn decay(&mut self, factor: f32) -> Result<(), CudaError> {
        let kernel = self
            .dev
            .get_func(MODULE_NAME, DECAY_KERNEL)
            .ok_or(CudaError::missing_function(DECAY_KERNEL))?;
        unsafe { kernel.launch(kernel_config(), (&mut self.tensor, factor)) }
            .map_err(|error| CudaError::new("tensor_quantale_decay", error))
    }

    pub fn seed_exploration(&mut self, engine: &mut ExplorationEngine) -> Result<(), CudaError> {
        let strategy_nodes = engine.strategy_nodes()?;
        let strategy_biases = engine.strategy_biases();
        let receipt_priors = engine.receipt_prior_vector();
        let strategy_count = i32::try_from(strategy_nodes.len())
            .map_err(|_| CudaError::invalid_input("too many exploration strategies"))?;
        let strategy_node_buffer = self
            .dev
            .htod_copy(strategy_nodes)
            .map_err(|error| CudaError::new("htod_copy exploration strategy nodes", error))?;
        let strategy_bias_buffer = self
            .dev
            .htod_copy(strategy_biases)
            .map_err(|error| CudaError::new("htod_copy exploration strategy bias", error))?;
        let receipt_prior_buffer = self
            .dev
            .htod_copy(receipt_priors)
            .map_err(|error| CudaError::new("htod_copy exploration receipt priors", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, EXPLORATION_SEED_KERNEL)
            .ok_or(CudaError::missing_function(EXPLORATION_SEED_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &self.tensor,
                    &strategy_node_buffer,
                    &strategy_bias_buffer,
                    &receipt_prior_buffer,
                    strategy_count,
                    EXPLORATION_MAX_TOKENS as i32,
                    &mut self.exploration_tokens,
                    &mut self.exploration_scores,
                    &mut self.exploration_parents,
                    &mut self.exploration_token_count,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_seed_exploration", error))?;
        self.sync_exploration_engine(engine)
    }

    pub fn expand_exploration(
        &mut self,
        engine: &mut ExplorationEngine,
    ) -> Result<Vec<ExplorationCandidate>, CudaError> {
        self.seed_exploration(engine)?;
        let max_depth = i32::try_from(engine.config().max_depth)
            .map_err(|_| CudaError::invalid_input("exploration max_depth too large"))?;
        let beam_width = i32::try_from(engine.config().beam_width)
            .map_err(|_| CudaError::invalid_input("exploration beam_width too large"))?;
        for source_depth in 0..max_depth {
            let expand = self
                .dev
                .get_func(MODULE_NAME, EXPLORATION_EXPAND_KERNEL)
                .ok_or(CudaError::missing_function(EXPLORATION_EXPAND_KERNEL))?;
            unsafe {
                expand.launch(
                    kernel_config(),
                    (
                        &self.tensor,
                        &mut self.exploration_token_count,
                        source_depth,
                        max_depth,
                        EXPLORATION_MAX_TOKENS as i32,
                        &mut self.exploration_tokens,
                        &mut self.exploration_parents,
                    ),
                )
            }
            .map_err(|error| CudaError::new("tensor_quantale_expand_tokens", error))?;
        }
        let score = self
            .dev
            .get_func(MODULE_NAME, EXPLORATION_SCORE_KERNEL)
            .ok_or(CudaError::missing_function(EXPLORATION_SCORE_KERNEL))?;
        unsafe {
            score.launch(
                kernel_config(),
                (
                    &self.exploration_tokens,
                    &self.exploration_token_count,
                    engine.config().novelty_weight,
                    engine.config().receipt_weight,
                    engine.config().entropy_penalty,
                    &mut self.exploration_scores,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_score_tokens", error))?;
        let terminal_visits = self
            .dev
            .htod_copy(engine.terminal_visit_vector())
            .map_err(|error| CudaError::new("htod_copy exploration terminal visits", error))?;
        let first_hop_visits = self
            .dev
            .htod_copy(engine.first_hop_visit_vector())
            .map_err(|error| CudaError::new("htod_copy exploration first-hop visits", error))?;
        let max_terminal_visits = i32::try_from(engine.config().max_terminal_visits)
            .map_err(|_| CudaError::invalid_input("exploration max_terminal_visits too large"))?;
        let max_first_hop_visits = i32::try_from(engine.config().max_first_hop_visits)
            .map_err(|_| CudaError::invalid_input("exploration max_first_hop_visits too large"))?;
        let topk = self
            .dev
            .get_func(MODULE_NAME, EXPLORATION_TOPK_KERNEL)
            .ok_or(CudaError::missing_function(EXPLORATION_TOPK_KERNEL))?;
        unsafe {
            topk.launch(
                kernel_config(),
                (
                    &self.exploration_tokens,
                    &self.exploration_scores,
                    &self.exploration_token_count,
                    beam_width,
                    engine.config().repeat_penalty,
                    max_terminal_visits,
                    max_first_hop_visits,
                    &terminal_visits,
                    &first_hop_visits,
                    &mut self.exploration_selected,
                    &mut self.exploration_selected_count,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_select_topk_tokens", error))?;
        self.sync_exploration_engine(engine)?;
        Ok(engine.selected().to_vec())
    }

    pub fn commit_exploration_candidate(
        &mut self,
        candidate: &ExplorationCandidate,
    ) -> Result<DecisionReport, CudaError> {
        let candidate_buffer = self
            .dev
            .htod_copy(vec![*candidate])
            .map_err(|error| CudaError::new("htod_copy exploration commit candidate", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, EXPLORATION_COMMIT_KERNEL)
            .ok_or(CudaError::missing_function(EXPLORATION_COMMIT_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (
                    &mut self.consumed,
                    &mut self.active,
                    &mut self.next_active,
                    &candidate_buffer,
                    &mut self.decision,
                ),
            )
        }
        .map_err(|error| CudaError::new("tensor_quantale_commit_exploration", error))?;
        self.decision_report()
    }

    /// Score dynamically detected JIT chains on the GPU and embed results into the tensor.
    pub fn embed_jit_chain_scores(
        &mut self,
        chains: &[crate::jit_kernel_fusion::JitChainMetadata],
        src_node: i32,
    ) -> Result<(), CudaError> {
        if chains.is_empty() {
            return Ok(());
        }
        let count = i32::try_from(chains.len())
            .map_err(|_| CudaError::invalid_input("too many JIT chains"))?;
        let chain_buf = self
            .dev
            .htod_copy(chains.to_vec())
            .map_err(|error| CudaError::new("htod_copy jit chain metadata", error))?;
        let kernel = self
            .dev
            .get_func(MODULE_NAME, JIT_CHAIN_SCORE_KERNEL)
            .ok_or(CudaError::missing_function(JIT_CHAIN_SCORE_KERNEL))?;
        unsafe {
            kernel.launch(
                kernel_config(),
                (&mut self.tensor, &chain_buf, count, src_node),
            )
        }
        .map_err(|error| CudaError::new("jit_chain_score_embed", error))?;
        Ok(())
    }

    fn sync_exploration_engine(&self, engine: &mut ExplorationEngine) -> Result<(), CudaError> {
        let token_count = self
            .dev
            .dtoh_sync_copy(&self.exploration_token_count)
            .map_err(|error| CudaError::new("dtoh_sync_copy exploration token_count", error))?
            .into_iter()
            .next()
            .unwrap_or(0)
            .clamp(0, EXPLORATION_MAX_TOKENS as i32) as usize;
        let selected_count = self
            .dev
            .dtoh_sync_copy(&self.exploration_selected_count)
            .map_err(|error| CudaError::new("dtoh_sync_copy exploration selected_count", error))?
            .into_iter()
            .next()
            .unwrap_or(0)
            .clamp(0, EXPLORATION_MAX_SELECTED as i32) as usize;
        let mut tokens = self
            .dev
            .dtoh_sync_copy(&self.exploration_tokens)
            .map_err(|error| CudaError::new("dtoh_sync_copy exploration tokens", error))?;
        let mut selected = self
            .dev
            .dtoh_sync_copy(&self.exploration_selected)
            .map_err(|error| CudaError::new("dtoh_sync_copy exploration selected", error))?;
        tokens.truncate(token_count);
        selected.truncate(selected_count);
        engine.load_gpu_state(tokens, selected);
        Ok(())
    }
}
