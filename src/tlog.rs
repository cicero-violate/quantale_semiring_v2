//! Append-only JSONL trace log for host-side quantale events.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::batch::BatchPlan;
use crate::exploration::ExplorationCommitRecord;
use crate::projection::DecisionReport;
use crate::rule_delta::{ExecutionReceipt, ProcessReceipt};
use crate::tensor::TensorEdge;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlogRecordKind {
    Decision,
    Receipt,
    TensorEdges,
    AgentStep,
    ExplorationSeed,
    ExplorationExpand,
    ExplorationTopK,
    ExplorationCommit,
    ExplorationReceipt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TlogRecordMeta {
    pub kind: TlogRecordKind,
    pub flags: u16,
    pub payload_len: u32,
    pub sequence: u64,
    pub checksum: u64,
}

#[derive(Serialize, Deserialize)]
struct JsonRecord<T> {
    sequence: u64,
    kind: TlogRecordKind,
    payload: T,
}

#[derive(Deserialize)]
struct JsonMetaRecord {
    sequence: u64,
    kind: TlogRecordKind,
}

pub struct TlogWriter {
    writer: BufWriter<File>,
    next_sequence: u64,
}

impl TlogWriter {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let next_sequence = count_jsonl_records(path.as_ref())?;
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            next_sequence,
        })
    }

    pub fn append_decision(&mut self, report: &DecisionReport) -> io::Result<u64> {
        self.append_record(TlogRecordKind::Decision, report)
    }

    pub fn append_receipt(&mut self, receipt: &ExecutionReceipt) -> io::Result<u64> {
        self.append_record(TlogRecordKind::Receipt, &json!(receipt))
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
                "blocked": decision.blocked,
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

    pub fn append_batch_plan(&mut self, label: &str, batch_plan: &BatchPlan) -> io::Result<u64> {
        self.append_record(
            TlogRecordKind::Decision,
            &json!({ "label": label, "batch_plan": batch_plan }),
        )
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
        serde_json::to_writer(&mut self.writer, &record)?;
        self.writer.write_all(b"\n")?;
        self.next_sequence += 1;
        Ok(sequence)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

pub fn read_record_meta(path: impl AsRef<Path>) -> io::Result<Vec<TlogRecordMeta>> {
    let file = File::open(path)?;
    let mut records = Vec::new();
    for (fallback_sequence, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let payload_len = u32::try_from(line.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "tlog JSON line is too large")
        })?;
        let parsed: JsonMetaRecord = serde_json::from_str(&line)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        records.push(TlogRecordMeta {
            kind: parsed.kind,
            flags: 0,
            payload_len,
            sequence: parsed.sequence.max(fallback_sequence as u64),
            checksum: payload_len as u64,
        });
    }
    Ok(records)
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
