//! pegastore — placement-aware immutable large-object cache for AI workloads.
//!
//! Objects are immutable and `put` is write-once. A value is a set of
//! **slots**; each slot is placed, tracked and evicted independently on a
//! `Device` (`Cpu { numa }` / `Gpu { index }`). Bytes move only through
//! `Iov`s into registered `MemoryRegion`s, so the store never owns the
//! caller's memory and never copies where it doesn't have to.
//!
//! Two layers:
//! - [`Store`]: the user handle (`put` / `get` / `get_many` / `stat` /
//!   `publish` / `remove` / `remove_prefix`), with builder forms.
//! - [`raw::Access`]: what a backend implements. `Memory` is the semantic
//!   reference; `Local` and `Remote` backends must match its behavior.

#![deny(unsafe_op_in_unsafe_fn)]

mod access;
mod error;
mod layer;
mod ops;
mod region;
pub mod services;
mod store;
mod types;

pub use error::{Error, ErrorKind, ErrorStatus, Result};
pub use ops::{OpGet, OpPublish, OpPut, PutSlot, RpGet, RpPut};
pub use region::{Iov, MemoryRegion};
pub use store::{FutureGet, FuturePut, Store};
pub use types::{
    AccessInfo, Capability, Device, Key, Location, NodeId, ObjectInfo, ObjectSpec, Placement,
    Replica, Retention, SlotIdx, SlotInfo, SlotSpec, Tier,
};

/// Implementer-facing surface: the backend trait, its dyn mirror, layers,
/// and validation helpers.
pub mod raw {
    pub use crate::access::{Access, AccessDyn, BoxedFuture, Servicer};
    pub use crate::layer::Layer;
    pub use crate::ops::validate_iovs;
}
