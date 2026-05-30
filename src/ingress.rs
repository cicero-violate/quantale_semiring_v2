//! Non-blocking inbound event queue for external producers.

use std::sync::mpsc::{Receiver, SendError, Sender, TryRecvError, channel};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InboundEvent {
    pub source: String,
    pub event_name: String,
    pub payload: Vec<u8>,
}

impl InboundEvent {
    pub fn new(
        source: impl Into<String>,
        event_name: impl Into<String>,
        payload: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            source: source.into(),
            event_name: event_name.into(),
            payload: payload.into(),
        }
    }
}

#[derive(Clone)]
pub struct IngressServer {
    sender: Sender<InboundEvent>,
}

impl IngressServer {
    pub fn new() -> (Self, Receiver<InboundEvent>) {
        let (sender, receiver) = channel();
        (Self { sender }, receiver)
    }

    pub fn push_event(&self, event: InboundEvent) -> Result<(), SendError<InboundEvent>> {
        self.sender.send(event)
    }
}

pub fn drain_available(receiver: &Receiver<InboundEvent>, limit: usize) -> Vec<InboundEvent> {
    let mut events = Vec::new();
    for _ in 0..limit {
        match receiver.try_recv() {
            Ok(event) => events.push(event),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
        }
    }
    events
}
