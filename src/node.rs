//! Unified state/control/event node universe.

pub const STATE_NODE_COUNT: usize = 13;
pub const CONTROL_NODE_COUNT: usize = 13;
pub const EVENT_NODE_COUNT: usize = 18;

pub const STATE_OFFSET: usize = 0;
pub const CONTROL_OFFSET: usize = STATE_OFFSET + STATE_NODE_COUNT;
pub const EVENT_OFFSET: usize = CONTROL_OFFSET + CONTROL_NODE_COUNT;

/// Unified matrix universe:
/// N = StateNode ⊔ ControlNode ⊔ EventNode.
pub const NODE_COUNT: usize = STATE_NODE_COUNT + CONTROL_NODE_COUNT + EVENT_NODE_COUNT;

/// Backwards-compatible name for older callers. The matrix no longer contains
/// only states; it contains all node kinds.
pub const STATE_COUNT: usize = NODE_COUNT;
pub const MATRIX_LEN: usize = NODE_COUNT * NODE_COUNT;
pub const THREAD_COUNT: usize = 512;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateNode {
    Goal = 0,
    Input = 1,
    Parse = 2,
    Map = 3,
    Search = 4,
    Score = 5,
    Select = 6,
    Plan = 7,
    Optimize = 8,
    Execute = 9,
    Validate = 10,
    Memory = 11,
    Learn = 12,
}

impl StateNode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Goal => "State::Goal",
            Self::Input => "State::Input",
            Self::Parse => "State::Parse",
            Self::Map => "State::Map",
            Self::Search => "State::Search",
            Self::Score => "State::Score",
            Self::Select => "State::Select",
            Self::Plan => "State::Plan",
            Self::Optimize => "State::Optimize",
            Self::Execute => "State::Execute",
            Self::Validate => "State::Validate",
            Self::Memory => "State::Memory",
            Self::Learn => "State::Learn",
        }
    }

    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Goal),
            1 => Some(Self::Input),
            2 => Some(Self::Parse),
            3 => Some(Self::Map),
            4 => Some(Self::Search),
            5 => Some(Self::Score),
            6 => Some(Self::Select),
            7 => Some(Self::Plan),
            8 => Some(Self::Optimize),
            9 => Some(Self::Execute),
            10 => Some(Self::Validate),
            11 => Some(Self::Memory),
            12 => Some(Self::Learn),
            _ => None,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlNode {
    Allow = 0,
    Block = 1,
    Retry = 2,
    Repair = 3,
    Commit = 4,
    Rollback = 5,
    Halt = 6,
    GateInput = 7,
    GateExecution = 8,
    GateReceipt = 9,
    GateMemory = 10,
    GateLearn = 11,
    ChooseBest = 12,
}

impl ControlNode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Allow => "Control::Allow",
            Self::Block => "Control::Block",
            Self::Retry => "Control::Retry",
            Self::Repair => "Control::Repair",
            Self::Commit => "Control::Commit",
            Self::Rollback => "Control::Rollback",
            Self::Halt => "Control::Halt",
            Self::GateInput => "Control::GateInput",
            Self::GateExecution => "Control::GateExecution",
            Self::GateReceipt => "Control::GateReceipt",
            Self::GateMemory => "Control::GateMemory",
            Self::GateLearn => "Control::GateLearn",
            Self::ChooseBest => "Control::ChooseBest",
        }
    }

    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Allow),
            1 => Some(Self::Block),
            2 => Some(Self::Retry),
            3 => Some(Self::Repair),
            4 => Some(Self::Commit),
            5 => Some(Self::Rollback),
            6 => Some(Self::Halt),
            7 => Some(Self::GateInput),
            8 => Some(Self::GateExecution),
            9 => Some(Self::GateReceipt),
            10 => Some(Self::GateMemory),
            11 => Some(Self::GateLearn),
            12 => Some(Self::ChooseBest),
            _ => None,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventNode {
    FactArrived = 0,
    InputAccepted = 1,
    ParseOk = 2,
    ParseErr = 3,
    MapReady = 4,
    CandidateFound = 5,
    ScoreReady = 6,
    TopKSelected = 7,
    PlanReady = 8,
    OptimizeReady = 9,
    ExecuteStarted = 10,
    ExecuteFinished = 11,
    ReceiptAttached = 12,
    ReceiptAccepted = 13,
    ReceiptRejected = 14,
    HashNonzero = 15,
    MemoryWritten = 16,
    LearnUpdated = 17,
}

impl EventNode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::FactArrived => "Event::FactArrived",
            Self::InputAccepted => "Event::InputAccepted",
            Self::ParseOk => "Event::ParseOk",
            Self::ParseErr => "Event::ParseErr",
            Self::MapReady => "Event::MapReady",
            Self::CandidateFound => "Event::CandidateFound",
            Self::ScoreReady => "Event::ScoreReady",
            Self::TopKSelected => "Event::TopKSelected",
            Self::PlanReady => "Event::PlanReady",
            Self::OptimizeReady => "Event::OptimizeReady",
            Self::ExecuteStarted => "Event::ExecuteStarted",
            Self::ExecuteFinished => "Event::ExecuteFinished",
            Self::ReceiptAttached => "Event::ReceiptAttached",
            Self::ReceiptAccepted => "Event::ReceiptAccepted",
            Self::ReceiptRejected => "Event::ReceiptRejected",
            Self::HashNonzero => "Event::HashNonzero",
            Self::MemoryWritten => "Event::MemoryWritten",
            Self::LearnUpdated => "Event::LearnUpdated",
        }
    }

    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::FactArrived),
            1 => Some(Self::InputAccepted),
            2 => Some(Self::ParseOk),
            3 => Some(Self::ParseErr),
            4 => Some(Self::MapReady),
            5 => Some(Self::CandidateFound),
            6 => Some(Self::ScoreReady),
            7 => Some(Self::TopKSelected),
            8 => Some(Self::PlanReady),
            9 => Some(Self::OptimizeReady),
            10 => Some(Self::ExecuteStarted),
            11 => Some(Self::ExecuteFinished),
            12 => Some(Self::ReceiptAttached),
            13 => Some(Self::ReceiptAccepted),
            14 => Some(Self::ReceiptRejected),
            15 => Some(Self::HashNonzero),
            16 => Some(Self::MemoryWritten),
            17 => Some(Self::LearnUpdated),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    State(StateNode),
    Control(ControlNode),
    Event(EventNode),
}

impl Node {
    pub const fn encode(self) -> i32 {
        match self {
            Self::State(state) => STATE_OFFSET as i32 + state as i32,
            Self::Control(control) => CONTROL_OFFSET as i32 + control as i32,
            Self::Event(event) => EVENT_OFFSET as i32 + event as i32,
        }
    }

    pub const fn decode(id: i32) -> Option<Self> {
        if id < 0 || id as usize >= NODE_COUNT {
            return None;
        }

        let raw = id as usize;
        if raw < CONTROL_OFFSET {
            return match StateNode::from_u32((raw - STATE_OFFSET) as u32) {
                Some(value) => Some(Self::State(value)),
                None => None,
            };
        }
        if raw < EVENT_OFFSET {
            return match ControlNode::from_u32((raw - CONTROL_OFFSET) as u32) {
                Some(value) => Some(Self::Control(value)),
                None => None,
            };
        }
        match EventNode::from_u32((raw - EVENT_OFFSET) as u32) {
            Some(value) => Some(Self::Event(value)),
            None => None,
        }
    }

    pub const fn state(state: StateNode) -> Self {
        Self::State(state)
    }

    pub const fn control(control: ControlNode) -> Self {
        Self::Control(control)
    }

    pub const fn event(event: EventNode) -> Self {
        Self::Event(event)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::State(value) => value.name(),
            Self::Control(value) => value.name(),
            Self::Event(value) => value.name(),
        }
    }
}

pub const START_NODE: Node = Node::State(StateNode::Goal);
pub const EXECUTE_PROBE_NODE: Node = Node::State(StateNode::Execute);
pub const LEARN_PROBE_NODE: Node = Node::State(StateNode::Learn);

pub fn node_name(node: i32) -> String {
    Node::decode(node).map_or_else(
        || format!("Unknown({node})"),
        |node| node.name().to_string(),
    )
}

/// Backwards-compatible name for older examples.
pub fn state_name(state: i32) -> String {
    node_name(state)
}
