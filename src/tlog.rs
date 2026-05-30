//! Append-only binary trace log for host-side quantale events.
//!
//! CUDA kernels compute reports and decisions. The Rust host appends those
//! compact records here; CUDA never writes the tlog directly.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::edge::TransitionEdge;
use crate::projection::{DecisionReport, QuantaleCudaReport};
use crate::receipt::ExecutionReceipt;

pub const TLOG_MAGIC: [u8; 8] = *b"QSV2TLOG";
pub const TLOG_VERSION: u16 = 1;
pub const TLOG_HEADER_LEN: u16 = 16;
pub const TLOG_RECORD_HEADER_LEN: usize = 24;

const TLOG_ENDIAN_LE: u16 = 0x0102;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlogRecordKind {
    Decision = 1,
    CudaReport = 2,
    Receipt = 3,
    TransitionEdges = 4,
}

impl TlogRecordKind {
    pub const fn from_u16(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::Decision),
            2 => Some(Self::CudaReport),
            3 => Some(Self::Receipt),
            4 => Some(Self::TransitionEdges),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TlogRecordMeta {
    pub kind: TlogRecordKind,
    pub flags: u16,
    pub payload_len: u32,
    pub sequence: u64,
    pub checksum: u64,
}

pub struct TlogWriter {
    file: File,
    next_sequence: u64,
}

impl TlogWriter {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)?;

        let len = file.metadata()?.len();
        if len == 0 {
            write_file_header(&mut file)?;
        } else {
            validate_file_header(&mut file)?;
        }

        let next_sequence = count_records(&mut file)?;
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            next_sequence,
        })
    }

    pub fn append_decision(&mut self, report: &DecisionReport) -> io::Result<u64> {
        let mut payload = Vec::with_capacity(28);
        push_i32(&mut payload, report.step);
        push_i32(&mut payload, report.selected_src);
        push_i32(&mut payload, report.selected_dst);
        push_i32(&mut payload, report.first_hop);
        push_f32(&mut payload, report.selected_value);
        push_i32(&mut payload, report.halted);
        push_i32(&mut payload, report.blocked);
        self.append_record(TlogRecordKind::Decision, &payload)
    }

    pub fn append_cuda_report(&mut self, report: &QuantaleCudaReport) -> io::Result<u64> {
        let mut payload = Vec::with_capacity(28);
        push_i32(&mut payload, report.step);
        push_i32(&mut payload, report.best_src);
        push_i32(&mut payload, report.best_dst);
        push_f32(&mut payload, report.best_value);
        push_i32(&mut payload, report.event_count);
        push_f32(&mut payload, report.goal_to_execute);
        push_f32(&mut payload, report.goal_to_learn);
        self.append_record(TlogRecordKind::CudaReport, &payload)
    }

    pub fn append_receipt(&mut self, receipt: &ExecutionReceipt) -> io::Result<u64> {
        let mut payload = Vec::with_capacity(28);
        payload.push(u8::from(receipt.accepted));
        payload.push(u8::from(receipt.hash_nonzero));
        payload.extend_from_slice(&[0, 0]);
        push_f32(&mut payload, receipt.receipt_confidence);
        push_f32(&mut payload, receipt.hash_score);
        push_f32(&mut payload, receipt.validation_score);
        push_f32(&mut payload, receipt.rejection_score);
        push_f32(&mut payload, receipt.rollback_score);
        push_f32(&mut payload, receipt.repair_score);
        self.append_record(TlogRecordKind::Receipt, &payload)
    }

    pub fn append_edges(&mut self, label: &str, edges: &[TransitionEdge]) -> io::Result<u64> {
        let label_bytes = label.as_bytes();
        let label_len = u32::try_from(label_bytes.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "tlog edge label is too long")
        })?;
        let edge_count = u32::try_from(edges.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "too many tlog transition edges",
            )
        })?;

        let mut payload = Vec::with_capacity(8 + label_bytes.len() + edges.len() * 12);
        push_u32(&mut payload, label_len);
        payload.extend_from_slice(label_bytes);
        push_u32(&mut payload, edge_count);
        for edge in edges {
            push_i32(&mut payload, edge.src);
            push_i32(&mut payload, edge.dst);
            push_f32(&mut payload, edge.value);
        }
        self.append_record(TlogRecordKind::TransitionEdges, &payload)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }

    fn append_record(&mut self, kind: TlogRecordKind, payload: &[u8]) -> io::Result<u64> {
        let sequence = self.next_sequence;
        let payload_len = u32::try_from(payload.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "tlog payload is too large")
        })?;
        let flags = 0_u16;
        let checksum = record_checksum(kind as u16, flags, payload_len, sequence, payload);

        let mut header = Vec::with_capacity(TLOG_RECORD_HEADER_LEN);
        push_u16(&mut header, kind as u16);
        push_u16(&mut header, flags);
        push_u32(&mut header, payload_len);
        push_u64(&mut header, sequence);
        push_u64(&mut header, checksum);

        self.file.write_all(&header)?;
        self.file.write_all(payload)?;
        self.next_sequence += 1;
        Ok(sequence)
    }
}

pub fn read_record_meta(path: impl AsRef<Path>) -> io::Result<Vec<TlogRecordMeta>> {
    let mut file = File::open(path)?;
    validate_file_header(&mut file)?;
    let mut records = Vec::new();

    loop {
        let mut header = [0_u8; TLOG_RECORD_HEADER_LEN];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }

        let kind_raw = u16::from_le_bytes([header[0], header[1]]);
        let kind = TlogRecordKind::from_u16(kind_raw).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid tlog record kind")
        })?;
        let flags = u16::from_le_bytes([header[2], header[3]]);
        let payload_len = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let sequence = u64::from_le_bytes([
            header[8], header[9], header[10], header[11], header[12], header[13], header[14],
            header[15],
        ]);
        let checksum = u64::from_le_bytes([
            header[16], header[17], header[18], header[19], header[20], header[21], header[22],
            header[23],
        ]);

        let mut payload = vec![0_u8; payload_len as usize];
        file.read_exact(&mut payload)?;
        let actual = record_checksum(kind_raw, flags, payload_len, sequence, &payload);
        if actual != checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tlog record checksum mismatch",
            ));
        }

        records.push(TlogRecordMeta {
            kind,
            flags,
            payload_len,
            sequence,
            checksum,
        });
    }

    Ok(records)
}

fn write_file_header(file: &mut File) -> io::Result<()> {
    file.write_all(&TLOG_MAGIC)?;
    file.write_all(&TLOG_VERSION.to_le_bytes())?;
    file.write_all(&TLOG_HEADER_LEN.to_le_bytes())?;
    file.write_all(&TLOG_ENDIAN_LE.to_le_bytes())?;
    file.write_all(&0_u16.to_le_bytes())
}

fn validate_file_header(file: &mut File) -> io::Result<()> {
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0_u8; TLOG_HEADER_LEN as usize];
    file.read_exact(&mut header)?;
    if header[0..8] != TLOG_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid tlog magic",
        ));
    }
    let version = u16::from_le_bytes([header[8], header[9]]);
    let header_len = u16::from_le_bytes([header[10], header[11]]);
    let endian = u16::from_le_bytes([header[12], header[13]]);
    if version != TLOG_VERSION || header_len != TLOG_HEADER_LEN || endian != TLOG_ENDIAN_LE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported tlog header",
        ));
    }
    Ok(())
}

fn count_records(file: &mut File) -> io::Result<u64> {
    validate_file_header(file)?;
    let mut count = 0_u64;
    loop {
        let mut header = [0_u8; TLOG_RECORD_HEADER_LEN];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }
        let payload_len = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        file.seek(SeekFrom::Current(i64::from(payload_len)))?;
        count += 1;
    }
    file.seek(SeekFrom::End(0))?;
    Ok(count)
}

fn record_checksum(kind: u16, flags: u16, payload_len: u32, sequence: u64, payload: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    fn feed(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    feed(&mut hash, &kind.to_le_bytes());
    feed(&mut hash, &flags.to_le_bytes());
    feed(&mut hash, &payload_len.to_le_bytes());
    feed(&mut hash, &sequence.to_le_bytes());
    feed(&mut hash, payload);
    hash
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}
