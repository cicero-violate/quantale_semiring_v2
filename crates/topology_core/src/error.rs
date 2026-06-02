use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyError {
    pub message: String,
}

impl TopologyError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "topology error: {}", self.message)
    }
}

impl std::error::Error for TopologyError {}
