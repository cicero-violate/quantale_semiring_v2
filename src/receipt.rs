use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProcessReceipt {
    pub node_name: String,
    pub exit_code: i32,
    pub stdout_payload: String,
    pub stderr_payload: String,
}
