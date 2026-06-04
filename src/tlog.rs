//! Append-only JSONL trace log for host-side quantale events.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::exploration::ExplorationCommitRecord;
use crate::graph::DecisionReport;
use crate::tensor::TensorEdge;
use crate::types::ProcessReceipt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlogRecordKind {
    Decision,
    TensorEdges,
    AgentStep,
    ExplorationSeed,
    ExplorationExpand,
    ExplorationTopK,
    ExplorationCommit,
    ExplorationReceipt,
}

#[derive(Serialize, Deserialize)]
struct JsonRecord<T> {
    sequence: u64,
    kind: TlogRecordKind,
    payload: T,
}

pub struct TlogWriter {
    file: File,
    next_sequence: u64,
}

impl TlogWriter {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let next_sequence = count_jsonl_records(path)?;
        // O_APPEND: every write() extends the file atomically at the OS level.
        // Combined with formatting the entire record + newline in memory before
        // calling write_all(), this prevents interleaved records when multiple
        // agent instances write to the same tlog file concurrently.
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file,
            next_sequence,
        })
    }

    pub fn append_decision(&mut self, report: &DecisionReport) -> io::Result<u64> {
        self.append_record(TlogRecordKind::Decision, report)
    }

    /// Fused step log: process receipt outcome + the decision that triggered it.
    pub fn log_step(
        &mut self,
        receipt: &ProcessReceipt,
        decision: &DecisionReport,
    ) -> io::Result<u64> {
        self.append_record(
            TlogRecordKind::AgentStep,
            &json!({
                "step": decision.step,
                "selected_src": decision.selected_src,
                "selected_dst": decision.selected_dst,
                "selected_value": decision.selected_value,
                "halted": decision.halted,
                "blocked": if receipt.exit_code == 0 { decision.blocked } else { 1 },
                "decision_blocked": decision.blocked,
                "node": receipt.node_name,
                "exit_code": receipt.exit_code,
                "stdout_len": receipt.stdout_payload.len(),
                "stderr": receipt.stderr_payload,
            }),
        )
    }

    pub fn append_tensor_edges(&mut self, label: &str, edges: &[TensorEdge]) -> io::Result<u64> {
        self.append_record(
            TlogRecordKind::TensorEdges,
            &json!({ "label": label, "edges": edges }),
        )
    }

    pub fn append_exploration_seed<T: Serialize>(&mut self, payload: &T) -> io::Result<u64> {
        self.append_record(TlogRecordKind::ExplorationSeed, payload)
    }

    pub fn append_exploration_expand<T: Serialize>(&mut self, payload: &T) -> io::Result<u64> {
        self.append_record(TlogRecordKind::ExplorationExpand, payload)
    }

    pub fn append_exploration_topk<T: Serialize>(&mut self, payload: &T) -> io::Result<u64> {
        self.append_record(TlogRecordKind::ExplorationTopK, payload)
    }

    pub fn append_exploration_commit(
        &mut self,
        record: &ExplorationCommitRecord,
    ) -> io::Result<u64> {
        self.append_record(TlogRecordKind::ExplorationCommit, record)
    }

    pub fn append_exploration_receipt<T: Serialize>(&mut self, payload: &T) -> io::Result<u64> {
        self.append_record(TlogRecordKind::ExplorationReceipt, payload)
    }

    pub fn append_record<T: Serialize>(
        &mut self,
        kind: TlogRecordKind,
        payload: &T,
    ) -> io::Result<u64> {
        let sequence = self.next_sequence;
        let record = JsonRecord {
            sequence,
            kind,
            payload,
        };
        // Serialize the entire record into memory first, then append the
        // newline, so the file receives exactly one write_all() call per
        // record.  With O_APPEND this write is atomic at the OS level,
        // preventing partial or interleaved records in the tlog file.
        let mut line = serde_json::to_vec(&record)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        line.push(b'\n');
        self.file.write_all(&line)?;
        self.next_sequence += 1;
        Ok(sequence)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

fn count_jsonl_records(path: &Path) -> io::Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    Ok(BufReader::new(File::open(path)?)
        .lines()
        .filter(|line| line.as_ref().map_or(true, |value| !value.trim().is_empty()))
        .count() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DecisionReport;
    use crate::types::ProcessReceipt;

    #[test]
    fn log_step_marks_failed_receipts_as_blocked() {
        let path = std::env::temp_dir().join(format!(
            "quantale_tlog_test_{}_{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut tlog = TlogWriter::open(&path).unwrap();
        let receipt = ProcessReceipt {
            node_name: "Analysis::Return1".to_string(),
            exit_code: 1,
            stdout_payload: String::new(),
            stderr_payload: "requires the cuda feature".to_string(),
        };
        let decision = DecisionReport {
            step: 7,
            selected_src: 1,
            selected_dst: 2,
            first_hop: 2,
            selected_value: 0.9,
            halted: 0,
            blocked: 0,
        };

        tlog.log_step(&receipt, &decision).unwrap();
        tlog.flush().unwrap();
        let line = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let value: serde_json::Value = serde_json::from_str(line.trim()).unwrap();

        assert_eq!(value["payload"]["exit_code"], 1);
        assert_eq!(value["payload"]["blocked"], 1);
        assert_eq!(value["payload"]["decision_blocked"], 0);
    }
}
