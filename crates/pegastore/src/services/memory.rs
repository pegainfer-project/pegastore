//! In-process reference backend. Defines the semantics every other backend
//! must reproduce: write-once slots, per-device pools with FIFO eviction of
//! `Cache` objects, `Explicit` never evicted, distance-based source
//! selection, `External` replicas that vanish with their region.
//!
//! It is an oracle, not a fast path: one mutex, byte copies through `Vec`.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

use bytes::Bytes;

use crate::access::Access;
use crate::error::{Error, ErrorKind, Result};
use crate::ops::{OpGet, OpPublish, OpPut, RpGet, RpPut};
use crate::region::{Iov, MemoryRegion};
use crate::types::{
    AccessInfo, Capability, Device, Key, Location, NodeId, ObjectInfo, ObjectSpec, Placement,
    Replica, Retention, SlotInfo, Tier,
};

pub struct MemoryBuilder {
    node: NodeId,
    devices: Vec<(Device, Option<u64>)>,
    gpu_numa: HashMap<u16, u16>,
    capability: Capability,
}

impl Default for MemoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBuilder {
    pub fn new() -> Self {
        Self {
            node: NodeId(0),
            devices: Vec::new(),
            gpu_numa: HashMap::new(),
            capability: Capability::all(),
        }
    }

    pub fn node(mut self, node: NodeId) -> Self {
        self.node = node;
        self
    }

    /// Add a device with unlimited capacity.
    pub fn device(mut self, device: Device) -> Self {
        self.devices.push((device, None));
        self
    }

    /// Add a device whose pool holds at most `capacity` bytes.
    pub fn device_with_capacity(mut self, device: Device, capacity: u64) -> Self {
        self.devices.push((device, Some(capacity)));
        self
    }

    /// Declare which NUMA node a GPU hangs off; drives the distance function.
    pub fn gpu_affinity(mut self, gpu: u16, numa: u16) -> Self {
        self.gpu_numa.insert(gpu, numa);
        self
    }

    pub fn capability(mut self, capability: Capability) -> Self {
        self.capability = capability;
        self
    }

    pub fn build(self) -> Memory {
        let devices: Vec<(Device, Option<u64>)> = if self.devices.is_empty() {
            vec![(Device::cpu(0), None)]
        } else {
            self.devices
        };
        let info = AccessInfo {
            name: "memory",
            node: self.node,
            devices: devices.iter().map(|(d, _)| *d).collect(),
            capability: self.capability,
        };
        let pools = devices
            .iter()
            .map(|(d, cap)| {
                (
                    *d,
                    Pool {
                        capacity: *cap,
                        used: 0,
                        fifo: VecDeque::new(),
                    },
                )
            })
            .collect();
        Memory {
            inner: Arc::new(Inner {
                info: Arc::new(info),
                device_order: devices.iter().map(|(d, _)| *d).collect(),
                gpu_numa: self.gpu_numa,
                state: Mutex::new(State {
                    objects: BTreeMap::new(),
                    pools,
                }),
            }),
        }
    }
}

#[derive(Clone)]
pub struct Memory {
    inner: Arc<Inner>,
}

impl Memory {
    pub fn builder() -> MemoryBuilder {
        MemoryBuilder::new()
    }

    /// Bytes currently held in the pool for `device`.
    pub fn used(&self, device: Device) -> u64 {
        self.inner
            .state
            .lock()
            .unwrap()
            .pools
            .get(&device)
            .map_or(0, |p| p.used)
    }
}

struct Inner {
    info: Arc<AccessInfo>,
    device_order: Vec<Device>,
    gpu_numa: HashMap<u16, u16>,
    state: Mutex<State>,
}

struct State {
    objects: BTreeMap<Bytes, Object>,
    pools: HashMap<Device, Pool>,
}

struct Pool {
    capacity: Option<u64>,
    used: u64,
    /// Insertion order of owned replicas, for eviction. Entries may be stale.
    fifo: VecDeque<(Bytes, u32)>,
}

struct Object {
    spec: ObjectSpec,
    slots: Vec<Slot>,
}

#[derive(Default)]
struct Slot {
    replicas: Vec<ReplicaEntry>,
}

struct ReplicaEntry {
    device: Device,
    tier: Tier,
    misplaced: bool,
    data: ReplicaData,
}

enum ReplicaData {
    Owned(Arc<[u8]>),
    External { segs: Vec<(usize, u64)>, region_id: u64 },
}

impl Inner {
    /// Same ordering as `pegastore_cuda::Topology::distance`; this backend is
    /// the semantic oracle, so the two must agree.
    fn distance(&self, a: Device, b: Device) -> u32 {
        if a == b {
            return 0;
        }
        match (a, b) {
            (Device::Gpu { .. }, Device::Gpu { .. }) => 1, // NVLink
            (Device::Cpu { numa }, Device::Gpu { index }) | (Device::Gpu { index }, Device::Cpu { numa }) => {
                match self.gpu_numa.get(&index) {
                    Some(n) if *n == numa => 2, // C2C / local PCIe
                    Some(_) => 4,               // cross socket + PCIe
                    None => 3,
                }
            }
            (Device::Cpu { .. }, Device::Cpu { .. }) => 3, // cross socket
        }
    }

    fn location(&self, device: Device, tier: Tier) -> Location {
        Location {
            node: self.info.node,
            device,
            tier,
        }
    }

    /// Drop every `External` replica published from `region_id`.
    fn retire_region(&self, region_id: u64) {
        let mut st = self.state.lock().unwrap();
        for obj in st.objects.values_mut() {
            for slot in &mut obj.slots {
                slot.replicas.retain(|r| !matches!(&r.data, ReplicaData::External { region_id: id, .. } if *id == region_id));
            }
        }
    }
}

/// Handle stored inside a `MemoryRegion`; retires published replicas on drop.
struct RegionGuard {
    inner: Weak<Inner>,
    id: u64,
}

impl Drop for RegionGuard {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.retire_region(self.id);
        }
    }
}

impl State {
    /// Make room for `need` bytes on `device` by evicting `Cache` replicas in
    /// FIFO order, never touching `protect`. Returns false if impossible.
    fn make_room(&mut self, device: Device, need: u64, protect: &Bytes) -> bool {
        let Some(cap) = self.pools[&device].capacity else {
            return true;
        };
        if need > cap {
            return false;
        }
        let mut deferred = Vec::new();
        while self.pools[&device].used + need > cap {
            let Some((key, slot_idx)) = self.pools.get_mut(&device).unwrap().fifo.pop_front() else {
                break;
            };
            if key == *protect {
                deferred.push((key, slot_idx));
                continue;
            }
            let Some(obj) = self.objects.get_mut(&key) else {
                continue;
            };
            if obj.spec.retention != Retention::Cache {
                continue;
            }
            let slot = &mut obj.slots[slot_idx as usize];
            let mut freed = 0;
            slot.replicas.retain(|r| {
                if r.device == device
                    && let ReplicaData::Owned(b) = &r.data
                {
                    freed += b.len() as u64;
                    false
                } else {
                    true
                }
            });
            self.pools.get_mut(&device).unwrap().used -= freed;
            if obj.slots.iter().all(|s| s.replicas.is_empty()) {
                self.objects.remove(&key);
            }
        }
        let pool = self.pools.get_mut(&device).unwrap();
        for e in deferred.into_iter().rev() {
            pool.fifo.push_front(e);
        }
        pool.used + need <= cap
    }

    fn charge(&mut self, device: Device, key: &Bytes, slot: u32, len: u64) {
        let pool = self.pools.get_mut(&device).unwrap();
        pool.used += len;
        pool.fifo.push_back((key.clone(), slot));
    }

    fn release_object(&mut self, obj: &Object) {
        for slot in &obj.slots {
            for r in &slot.replicas {
                if let ReplicaData::Owned(b) = &r.data
                    && let Some(pool) = self.pools.get_mut(&r.device)
                {
                    pool.used -= b.len() as u64;
                }
            }
        }
    }
}

// SAFETY helpers: the caller registered these regions and promised validity.
unsafe fn gather(iovs: &[Iov<'_>]) -> Vec<u8> {
    let total: usize = iovs.iter().map(|i| i.len as usize).sum();
    let mut out = Vec::with_capacity(total);
    for iov in iovs {
        // SAFETY: bounds were validated; region memory is host-accessible for Memory.
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(iov.as_ptr(), iov.len as usize) });
    }
    out
}

unsafe fn scatter(src: &[u8], iovs: &[Iov<'_>]) {
    let mut off = 0usize;
    for iov in iovs {
        let n = iov.len as usize;
        // SAFETY: bounds validated by caller; src has >= off + n bytes.
        unsafe { std::ptr::copy_nonoverlapping(src[off..off + n].as_ptr(), iov.as_ptr(), n) };
        off += n;
    }
}

impl ReplicaData {
    fn len(&self) -> u64 {
        match self {
            ReplicaData::Owned(b) => b.len() as u64,
            ReplicaData::External { segs, .. } => segs.iter().map(|(_, l)| l).sum(),
        }
    }

    /// Copy `[offset, offset + len)` of this replica into a fresh Vec.
    fn read(&self, offset: u64, len: u64) -> Vec<u8> {
        match self {
            ReplicaData::Owned(b) => b[offset as usize..(offset + len) as usize].to_vec(),
            ReplicaData::External { segs, .. } => {
                let mut out = Vec::with_capacity(len as usize);
                let mut skip = offset;
                let mut want = len;
                for (addr, seg_len) in segs {
                    if want == 0 {
                        break;
                    }
                    if skip >= *seg_len {
                        skip -= seg_len;
                        continue;
                    }
                    let n = (seg_len - skip).min(want);
                    // SAFETY: published regions were validated and are kept alive by RegionGuard.
                    let s = unsafe {
                        std::slice::from_raw_parts((*addr + skip as usize) as *const u8, n as usize)
                    };
                    out.extend_from_slice(s);
                    skip = 0;
                    want -= n;
                }
                out
            }
        }
    }
}

fn unknown_device(device: Device) -> Error {
    Error::new(ErrorKind::InvalidInput, "device not served by this backend")
        .with_context("device", device)
}

impl Access for Memory {
    fn info(&self) -> Arc<AccessInfo> {
        self.inner.info.clone()
    }

    unsafe fn register(&self, ptr: *mut u8, len: u64, device: Device) -> Result<MemoryRegion> {
        if device.is_gpu() && !self.inner.info.capability.gpu_memory {
            return Err(Error::new(ErrorKind::Unsupported, "gpu memory disabled"));
        }
        let id = MemoryRegion::next_id();
        let guard = RegionGuard {
            inner: Arc::downgrade(&self.inner),
            id,
        };
        // SAFETY: the caller upholds MemoryRegion's contract.
        Ok(unsafe { MemoryRegion::with_id(id, ptr, len, device, Box::new(guard)) })
    }

    async fn put(&self, key: Key, op: OpPut<'_>) -> Result<RpPut> {
        op.validate()?;
        let spec = op.spec();
        let inner = &*self.inner;
        let mut st = inner.state.lock().unwrap();
        let kb = key.0.clone();

        // Phase 1: reserve the key; decide which slots are writable.
        let writable: Vec<bool> = match st.objects.get(&kb) {
            Some(obj) => {
                if obj.spec != spec {
                    return Err(Error::new(ErrorKind::SpecMismatch, "existing object has a different spec")
                        .with_context("key", format!("{key:?}")));
                }
                obj.slots.iter().map(|s| s.replicas.is_empty()).collect()
            }
            None => {
                st.objects.insert(
                    kb.clone(),
                    Object {
                        spec: spec.clone(),
                        slots: (0..spec.slots.len()).map(|_| Slot::default()).collect(),
                    },
                );
                vec![true; spec.slots.len()]
            }
        };

        // Phase 2: place each writable slot (may evict other objects).
        let mut results = Vec::with_capacity(op.slots.len());
        for (i, ps) in op.slots.iter().enumerate() {
            if !writable[i] {
                results.push(Err(Error::new(ErrorKind::AlreadyExists, "slot already written")
                    .with_context("slot", i)));
                continue;
            }
            let len = ps.spec.len;
            let candidates: Vec<(Device, bool)> = match &ps.spec.placement {
                Placement::Strict(d) => vec![(*d, false)],
                Placement::Prefer(d) => std::iter::once((*d, false))
                    .chain(inner.device_order.iter().filter(|x| *x != d).map(|x| (*x, true)))
                    .collect(),
                Placement::Each(ds) => ds.iter().map(|d| (*d, false)).collect(),
                Placement::Anywhere => inner.device_order.iter().map(|d| (*d, false)).collect(),
            };
            let each = matches!(ps.spec.placement, Placement::Each(_));

            let mut targets: Vec<(Device, bool)> = Vec::new();
            let mut err: Option<Error> = None;
            for (d, misplaced) in candidates {
                if !st.pools.contains_key(&d) {
                    err = Some(unknown_device(d));
                    if each {
                        break;
                    }
                    continue;
                }
                if st.make_room(d, len, &kb) {
                    targets.push((d, misplaced));
                    if !each {
                        break;
                    }
                } else {
                    err = Some(
                        Error::new(ErrorKind::NoSpace, "pool cannot fit slot")
                            .with_context("device", d)
                            .with_context("len", len),
                    );
                    if each {
                        break;
                    }
                }
            }
            let ok = if each { err.is_none() && !targets.is_empty() } else { !targets.is_empty() };
            if !ok {
                results.push(Err(err.unwrap_or_else(|| Error::new(ErrorKind::NoSpace, "no device available"))
                    .with_context("slot", i)));
                continue;
            }

            // SAFETY: validated iovs into caller-registered host memory.
            let bytes: Arc<[u8]> = unsafe { gather(ps.src) }.into();
            let obj = st.objects.get_mut(&kb).unwrap();
            for (d, misplaced) in &targets {
                obj.slots[i].replicas.push(ReplicaEntry {
                    device: *d,
                    tier: Tier::Dram,
                    misplaced: *misplaced,
                    data: ReplicaData::Owned(bytes.clone()),
                });
            }
            for (d, _) in &targets {
                st.charge(*d, &kb, i as u32, len);
            }
            results.push(Ok(()));
        }
        Ok(RpPut { slots: results })
    }

    async fn publish(&self, key: Key, op: OpPublish<'_>) -> Result<()> {
        let (device, total) = op.validate()?;
        let inner = &*self.inner;
        let mut st = inner.state.lock().unwrap();
        let obj = st
            .objects
            .get_mut(&key.0)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "no such key").with_context("key", format!("{key:?}")))?;
        let idx = op.slot.0 as usize;
        let spec = obj.spec.slots.get(idx).ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, "slot index out of range").with_context("slot", idx)
        })?;
        if spec.len != total {
            return Err(Error::new(ErrorKind::InvalidInput, "published length differs from slot length")
                .with_context("slot", idx)
                .with_context("len", total)
                .with_context("expected", spec.len));
        }
        let region_id = op.src[0].region.id();
        obj.slots[idx].replicas.push(ReplicaEntry {
            device,
            tier: Tier::External,
            misplaced: false,
            data: ReplicaData::External {
                segs: op.src.iter().map(|i| (i.as_ptr() as usize, i.len)).collect(),
                region_id,
            },
        });
        Ok(())
    }

    async fn stat(&self, keys: &[Key]) -> Result<Vec<Option<ObjectInfo>>> {
        let inner = &*self.inner;
        let st = inner.state.lock().unwrap();
        Ok(keys
            .iter()
            .map(|k| {
                st.objects.get(&k.0).map(|obj| ObjectInfo {
                    retention: obj.spec.retention,
                    slots: obj
                        .slots
                        .iter()
                        .zip(&obj.spec.slots)
                        .map(|(s, spec)| SlotInfo {
                            len: spec.len,
                            replicas: s
                                .replicas
                                .iter()
                                .map(|r| Replica {
                                    location: inner.location(r.device, r.tier),
                                    misplaced: r.misplaced,
                                })
                                .collect(),
                        })
                        .collect(),
                })
            })
            .collect())
    }

    async fn get(&self, key: Key, op: OpGet<'_>) -> Result<RpGet> {
        let (dst_device, dst_len) = op.validate()?;
        let inner = &*self.inner;
        let st = inner.state.lock().unwrap();
        let obj = st
            .objects
            .get(&key.0)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "no such key").with_context("key", format!("{key:?}")))?;
        let idx = op.slot.0 as usize;
        let spec = obj.spec.slots.get(idx).ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, "slot index out of range").with_context("slot", idx)
        })?;
        let end = op
            .src_offset
            .checked_add(dst_len)
            .filter(|e| *e <= spec.len)
            .ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "read range exceeds slot")
                    .with_context("src_offset", op.src_offset)
                    .with_context("len", dst_len)
                    .with_context("slot_len", spec.len)
            })?;
        debug_assert!(end <= spec.len);
        let slot = &obj.slots[idx];
        let replica = slot
            .replicas
            .iter()
            .min_by_key(|r| (inner.distance(dst_device, r.device), tier_rank(r.tier)))
            .ok_or_else(|| {
                Error::new(ErrorKind::Evicted, "slot has no replica")
                    .with_context("key", format!("{key:?}"))
                    .with_context("slot", idx)
            })?;
        debug_assert_eq!(replica.data.len(), spec.len);
        let bytes = replica.data.read(op.src_offset, dst_len);
        // SAFETY: dst iovs validated; Memory serves host-accessible regions only.
        unsafe { scatter(&bytes, op.dst) };
        Ok(RpGet {
            from: inner.location(replica.device, replica.tier),
        })
    }

    async fn remove(&self, keys: &[Key]) -> Result<()> {
        let inner = &*self.inner;
        let mut st = inner.state.lock().unwrap();
        for k in keys {
            if let Some(obj) = st.objects.remove(&k.0) {
                st.release_object(&obj);
            }
        }
        Ok(())
    }

    async fn remove_prefix(&self, prefix: &[u8]) -> Result<u64> {
        let inner = &*self.inner;
        let mut st = inner.state.lock().unwrap();
        let keys: Vec<Bytes> = st
            .objects
            .range(Bytes::copy_from_slice(prefix)..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, _)| k.clone())
            .collect();
        let n = keys.len() as u64;
        for k in keys {
            if let Some(obj) = st.objects.remove(&k) {
                st.release_object(&obj);
            }
        }
        Ok(n)
    }
}

fn tier_rank(t: Tier) -> u8 {
    match t {
        Tier::Dram => 0,
        Tier::External => 1,
        Tier::Ssd => 2,
    }
}
