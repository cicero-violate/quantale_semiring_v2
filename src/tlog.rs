//! Append-only JSONL trace log for host-side quantale events.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::edge::TransitionEdge;
use crate::projection::{DecisionReport, QuantaleCudaReport};
use crate::receipt::ExecutionReceipt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlogRecordKind {
    Decision,
    CudaReport,
    Receipt,
    TransitionEdges,
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

    pub fn append_cuda_report(&mut self, report: &QuantaleCudaReport) -> io::Result<u64> {
        self.append_record(TlogRecordKind::CudaReport, report)
    }

    pub fn append_receipt(&mut self, receipt: &ExecutionReceipt) -> io::Result<u64> {
        self.append_record(TlogRecordKind::Receipt, &json!(receipt))
    }

    pub fn append_edges(&mut self, label: &str, edges: &[TransitionEdge]) -> io::Result<u64> {
        self.append_record(
            TlogRecordKind::TransitionEdges,
            &json!({ "label": label, "edges": edges }),
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
