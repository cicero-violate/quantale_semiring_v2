//! Compact numeric node universe.
//!
//! Node identities are stable integer slots. Human names and graph structure are
//! owned by the data assets; this module only preserves the numeric ABI used by
//! CUDA kernels and existing callers.

pub const STATE_NODE_COUNT: usize = 13;
pub const CONTROL_NODE_COUNT: usize = 13;
pub const EVENT_NODE_COUNT: usize = 18;

pub const STATE_OFFSET: usize = 0;
pub const CONTROL_OFFSET: usize = STATE_OFFSET + STATE_NODE_COUNT;
pub const EVENT_OFFSET: usize = CONTROL_OFFSET + CONTROL_NODE_COUNT;
pub const NODE_COUNT: usize = STATE_NODE_COUNT + CONTROL_NODE_COUNT + EVENT_NODE_COUNT;

pub const STATE_COUNT: usize = NODE_COUNT;
pub const MATRIX_LEN: usize = NODE_COUNT * NODE_COUNT;
pub const THREAD_COUNT: usize = 512;

const NODE_NAMES: [&str; NODE_COUNT] = [
    "State::Goal",
    "State::Input",
    "State::Parse",
    "State::Map",
    "State::Search",
    "State::Score",
    "State::Select",
    "State::Plan",
    "State::Optimize",
    "State::Execute",
    "State::Validate",
    "State::Memory",
    "State::Learn",
    "Control::Allow",
    "Control::Block",
    "Control::Retry",
    "Control::Repair",
    "Control::Commit",
    "Control::Rollback",
    "Control::Halt",
    "Control::GateInput",
    "Control::GateExecution",
    "Control::GateReceipt",
    "Control::GateMemory",
    "Control::GateLearn",
    "Control::ChooseBest",
    "Event::FactArrived",
    "Event::InputAccepted",
    "Event::ParseOk",
    "Event::ParseErr",
    "Event::MapReady",
    "Event::CandidateFound",
    "Event::ScoreReady",
    "Event::TopKSelected",
    "Event::PlanReady",
    "Event::OptimizeReady",
    "Event::ExecuteStarted",
    "Event::ExecuteFinished",
    "Event::ReceiptAttached",
    "Event::ReceiptAccepted",
    "Event::ReceiptRejected",
    "Event::HashNonzero",
    "Event::MemoryWritten",
    "Event::LearnUpdated",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateNode(pub u32);

#[allow(non_upper_case_globals)]
impl StateNode {
    pub const Goal: Self = Self(0);
    pub const Input: Self = Self(1);
    pub const Parse: Self = Self(2);
    pub const Map: Self = Self(3);
    pub const Search: Self = Self(4);
    pub const Score: Self = Self(5);
    pub const Select: Self = Self(6);
    pub const Plan: Self = Self(7);
    pub const Optimize: Self = Self(8);
    pub const Execute: Self = Self(9);
    pub const Validate: Self = Self(10);
    pub const Memory: Self = Self(11);
    pub const Learn: Self = Self(12);

    pub fn name(self) -> &'static str {
        NODE_NAMES[STATE_OFFSET + self.0 as usize]
    }

    pub fn from_u32(value: u32) -> Option<Self> {
        ((value as usize) < STATE_NODE_COUNT).then_some(Self(value))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlNode(pub u32);

#[allow(non_upper_case_globals)]
impl ControlNode {
    pub const Allow: Self = Self(0);
    pub const Block: Self = Self(1);
    pub const Retry: Self = Self(2);
    pub const Repair: Self = Self(3);
    pub const Commit: Self = Self(4);
    pub const Rollback: Self = Self(5);
    pub const Halt: Self = Self(6);
    pub const GateInput: Self = Self(7);
    pub const GateExecution: Self = Self(8);
    pub const GateReceipt: Self = Self(9);
    pub const GateMemory: Self = Self(10);
    pub const GateLearn: Self = Self(11);
    pub const ChooseBest: Self = Self(12);

    pub fn name(self) -> &'static str {
        NODE_NAMES[CONTROL_OFFSET + self.0 as usize]
    }

    pub fn from_u32(value: u32) -> Option<Self> {
        ((value as usize) < CONTROL_NODE_COUNT).then_some(Self(value))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventNode(pub u32);

#[allow(non_upper_case_globals)]
impl EventNode {
    pub const FactArrived: Self = Self(0);
    pub const InputAccepted: Self = Self(1);
    pub const ParseOk: Self = Self(2);
    pub const ParseErr: Self = Self(3);
    pub const MapReady: Self = Self(4);
    pub const CandidateFound: Self = Self(5);
    pub const ScoreReady: Self = Self(6);
    pub const TopKSelected: Self = Self(7);
    pub const PlanReady: Self = Self(8);
    pub const OptimizeReady: Self = Self(9);
    pub const ExecuteStarted: Self = Self(10);
    pub const ExecuteFinished: Self = Self(11);
    pub const ReceiptAttached: Self = Self(12);
    pub const ReceiptAccepted: Self = Self(13);
    pub const ReceiptRejected: Self = Self(14);
    pub const HashNonzero: Self = Self(15);
    pub const MemoryWritten: Self = Self(16);
    pub const LearnUpdated: Self = Self(17);

    pub fn name(self) -> &'static str {
        NODE_NAMES[EVENT_OFFSET + self.0 as usize]
    }

    pub fn from_u32(value: u32) -> Option<Self> {
        ((value as usize) < EVENT_NODE_COUNT).then_some(Self(value))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Node(i32);

impl Node {
    pub const fn encode(self) -> i32 {
        self.0
    }

    pub fn decode(id: i32) -> Option<Self> {
        if id >= 0 && (id as usize) < NODE_COUNT {
            Some(Self(id))
        } else {
            None
        }
    }

    pub fn decode_index(raw: usize) -> Option<Self> {
        (raw < NODE_COUNT).then_some(Self(raw as i32))
    }

    pub const fn state(state: StateNode) -> Self {
        Self(STATE_OFFSET as i32 + state.0 as i32)
    }

    pub const fn control(control: ControlNode) -> Self {
        Self(CONTROL_OFFSET as i32 + control.0 as i32)
    }

    pub const fn event(event: EventNode) -> Self {
        Self(EVENT_OFFSET as i32 + event.0 as i32)
    }

    pub fn name(self) -> &'static str {
        NODE_NAMES[self.0 as usize]
    }
}

pub const START_NODE: Node = Node::state(StateNode::Goal);
pub const EXECUTE_PROBE_NODE: Node = Node::state(StateNode::Execute);
pub const LEARN_PROBE_NODE: Node = Node::state(StateNode::Learn);

pub fn node_name(node: i32) -> String {
    Node::decode(node).map_or_else(
        || format!("Unknown({node})"),
        |node| node.name().to_string(),
    )
}

pub fn state_name(state: i32) -> String {
    node_name(state)
}
