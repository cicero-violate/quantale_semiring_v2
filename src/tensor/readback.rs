use super::*;

impl TensorQuantaleWorld {
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.dev
    }

    pub fn tensor(&self) -> Result<Vec<f32>, CudaError> {
        self.dev
            .dtoh_sync_copy(&self.tensor)
            .map_err(|error| CudaError::new("dtoh_sync_copy tensor", error))
    }

    pub fn witness(&self) -> Result<Vec<i32>, CudaError> {
        self.dev
            .dtoh_sync_copy(&self.witness)
            .map_err(|error| CudaError::new("dtoh_sync_copy tensor witness", error))
    }

    pub fn reconstruct_tensor_path(
        &self,
        layer: i32,
        src: Node,
        dst: Node,
    ) -> Result<Vec<Node>, CudaError> {
        if !(0..TENSOR_LAYER_COUNT as i32).contains(&layer) {
            return Err(CudaError::invalid_input(format!(
                "invalid tensor layer {layer}"
            )));
        }
        let witness = self.witness()?;
        let offset = layer as usize * MATRIX_LEN;
        let registry = bundled_registry()?;
        reconstruct_path_from_witness_matrix(
            &witness[offset..offset + MATRIX_LEN],
            src,
            dst,
            &registry,
        )
    }

    pub fn reconstruct_projected_tensor_path(&self, layer: i32) -> Result<Vec<Node>, CudaError> {
        let decision = self.decision_report()?;
        let registry = bundled_registry()?;
        let src = Node::decode(decision.selected_src, &registry).ok_or_else(|| {
            CudaError::invalid_input(format!(
                "cannot reconstruct tensor path with invalid selected_src {}",
                decision.selected_src
            ))
        })?;
        let dst = Node::decode(decision.selected_dst, &registry).ok_or_else(|| {
            CudaError::invalid_input(format!(
                "cannot reconstruct tensor path with invalid selected_dst {}",
                decision.selected_dst
            ))
        })?;
        self.reconstruct_tensor_path(layer, src, dst)
    }

    pub fn decision_report(&self) -> Result<DecisionReport, CudaError> {
        let reports = self
            .dev
            .dtoh_sync_copy(&self.decision)
            .map_err(|error| CudaError::new("dtoh_sync_copy tensor decision", error))?;
        reports.into_iter().next().ok_or(CudaError {
            operation: "dtoh_sync_copy tensor decision",
            message: "empty tensor decision buffer".to_string(),
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.dev
            .synchronize()
            .map_err(|error| CudaError::new("CudaDevice::synchronize tensor", error))
    }
}
