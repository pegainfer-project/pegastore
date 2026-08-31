//! Core vocabulary: keys, slots, devices, placement, retention, metadata, capability.

use std::fmt;

use bytes::Bytes;

/// Opaque object key. The index is prefix-ordered, so callers encode namespaces
/// as key prefixes and reclaim them with `remove_prefix`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(pub Bytes);

impl Key {
    pub fn new(bytes: impl Into<Bytes>) -> Self {
        Self(bytes.into())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn starts_with(&self, prefix: &[u8]) -> bool {
        self.0.starts_with(prefix)
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match std::str::from_utf8(&self.0) {
            Ok(s) if s.chars().all(|c| !c.is_control()) => write!(f, "Key({s:?})"),
            _ => write!(f, "Key(0x{})", hex(&self.0)),
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl From<&str> for Key {
    fn from(s: &str) -> Self {
        Self(Bytes::copy_from_slice(s.as_bytes()))
    }
}
impl From<String> for Key {
    fn from(s: String) -> Self {
        Self(Bytes::from(s))
    }
}
impl From<Vec<u8>> for Key {
    fn from(v: Vec<u8>) -> Self {
        Self(Bytes::from(v))
    }
}
impl From<&[u8]> for Key {
    fn from(v: &[u8]) -> Self {
        Self(Bytes::copy_from_slice(v))
    }
}
impl From<Bytes> for Key {
    fn from(b: Bytes) -> Self {
        Self(b)
    }
}

/// Index of a slot inside an object. Slots are the unit of placement,
/// visibility and eviction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotIdx(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u64);

/// Physical position coordinate. The backend supplies the distance function.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Device {
    Cpu { numa: u16 },
    Gpu { index: u16 },
}

impl Device {
    pub const fn cpu(numa: u16) -> Self {
        Device::Cpu { numa }
    }

    pub const fn gpu(index: u16) -> Self {
        Device::Gpu { index }
    }

    pub const fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu { .. })
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu { numa } => write!(f, "cpu:{numa}"),
            Device::Gpu { index } => write!(f, "gpu:{index}"),
        }
    }
}

/// Who owns the bytes of a replica.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// Store-owned host memory.
    Dram,
    /// Store-owned SSD.
    Ssd,
    /// User memory registered via `publish`; lives as long as the user's region.
    External,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Location {
    pub node: NodeId,
    pub device: Device,
    pub tier: Tier,
}

/// Where a slot's bytes should land. Evaluated per slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Exactly this device; `NoSpace` otherwise.
    Strict(Device),
    /// This device if possible; otherwise anywhere, flagged `misplaced`.
    Prefer(Device),
    /// One replica on each listed device.
    Each(Vec<Device>),
    /// Wherever there is room.
    Anywhere,
}

/// Eviction class of an object.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Retention {
    /// May be evicted under pressure.
    Cache,
    /// Only removed by `remove`. A pool full of `Explicit` objects answers `NoSpace`.
    Explicit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotSpec {
    pub len: u64,
    pub placement: Placement,
}

impl SlotSpec {
    pub fn new(len: u64, placement: Placement) -> Self {
        Self { len, placement }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectSpec {
    pub retention: Retention,
    pub slots: Vec<SlotSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replica {
    pub location: Location,
    /// Placed somewhere other than the preferred device.
    pub misplaced: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotInfo {
    pub len: u64,
    /// Empty means the slot has no bytes anywhere (never landed, or evicted).
    pub replicas: Vec<Replica>,
}

impl SlotInfo {
    pub fn is_resident(&self) -> bool {
        !self.replicas.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectInfo {
    pub retention: Retention,
    pub slots: Vec<SlotInfo>,
}

impl ObjectInfo {
    pub fn resident_slots(&self) -> impl Iterator<Item = SlotIdx> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_resident())
            .map(|(i, _)| SlotIdx(i as u32))
    }

    pub fn is_fully_resident(&self) -> bool {
        self.slots.iter().all(SlotInfo::is_resident)
    }
}

/// What a backend can do. One flag per optional behavior; no implicit fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    /// `register` accepts `Device::Gpu`.
    pub gpu_memory: bool,
    pub publish: bool,
    pub placement_strict: bool,
    pub placement_each: bool,
    pub retention_explicit: bool,
    pub remove_prefix: bool,
    pub max_slot_len: Option<u64>,
    pub max_slots: Option<u32>,
    /// Maximum number of `Iov` segments per src/dst list.
    pub max_iov: Option<u32>,
    pub max_stat_batch: Option<u32>,
}

impl Capability {
    pub const fn all() -> Self {
        Self {
            gpu_memory: true,
            publish: true,
            placement_strict: true,
            placement_each: true,
            retention_explicit: true,
            remove_prefix: true,
            max_slot_len: None,
            max_slots: None,
            max_iov: None,
            max_stat_batch: None,
        }
    }

    pub const fn none() -> Self {
        Self {
            gpu_memory: false,
            publish: false,
            placement_strict: false,
            placement_each: false,
            retention_explicit: false,
            remove_prefix: false,
            max_slot_len: None,
            max_slots: None,
            max_iov: None,
            max_stat_batch: None,
        }
    }
}

impl Default for Capability {
    fn default() -> Self {
        Self::none()
    }
}

#[derive(Clone, Debug)]
pub struct AccessInfo {
    pub name: &'static str,
    pub node: NodeId,
    /// Devices this backend can place bytes on.
    pub devices: Vec<Device>,
    pub capability: Capability,
}
