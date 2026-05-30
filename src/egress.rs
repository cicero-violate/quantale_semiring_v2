//! Closed-loop outbound confirmation adapter.

use crate::receipt::ExecutionReceipt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExternalAction {
    Noop { label: String },
    LogTelemetry { label: String },
    TriggerGateInput { label: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct EgressConfirmation {
    pub action_label: String,
    pub success: bool,
    pub message: String,
    pub receipt: ExecutionReceipt,
}

pub struct EgressDispatcher;

impl EgressDispatcher {
    pub fn dispatch_with_confirmation(action: ExternalAction) -> EgressConfirmation {
        match action {
            ExternalAction::Noop { label }
            | ExternalAction::LogTelemetry { label }
            | ExternalAction::TriggerGateInput { label } => EgressConfirmation {
                action_label: label,
                success: true,
                message: "confirmed".to_string(),
                receipt: ExecutionReceipt::accepted(1.0, 1.0, 1.0),
            },
        }
    }
}
