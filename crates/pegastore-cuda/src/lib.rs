//! Single-node CUDA backend for pegastore.
//!
//! - DRAM tier: pinned pools, one per NUMA node, first-touched on that node.
//! - GPUs: copy endpoints for `get`/`put`, and external sources via `publish`.
//! - Source selection: `Topology::distance` (same device < NVLink / local C2C
//!   < cross socket).

#![deny(unsafe_op_in_unsafe_fn)]

pub mod cuda;
mod local;
mod pinned;
pub mod topology;

pub use local::{Local, LocalBuilder};
pub use pinned::{PinnedBuf, PinnedPool};
pub use topology::Topology;
