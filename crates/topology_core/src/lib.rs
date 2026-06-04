pub mod check;
pub mod compile;
pub mod error;
pub mod fusion;
pub mod model;
pub mod overlay;
pub mod programs;
pub mod registry;

pub use check::{
    DominatorPair, TopologyInvariants, TopologyViolation, ViolationKind, check,
    check_with_operators, format_violations,
};
pub use error::TopologyError;
pub use model::{
    CompiledTopology, CompiledTransition, GraphTopology, TopologyNode, TopologyPage,
    TopologyTransition,
};
pub use overlay::build_overlay_assets;
pub use registry::NodeRegistry;
