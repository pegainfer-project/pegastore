//! Single-node backend: per-NUMA pinned DRAM pools as the store-owned tier,
//! GPUs as copy endpoints and (via `publish`) as external sources.
//!
//! Copies are synchronous inside the async methods (`cuMemcpy*Async` +
//! stream sync); drive concurrent operations from separate threads. The
//! metadata lock is never held across a copy: readers clone an `Arc` to the
//! replica bytes, so eviction can drop its reference while a transfer is in
//! flight and the memory is returned only when the last reader is done.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

use bytes::Bytes;
use pegastore::raw::Access;
use pegastore::{
    AccessInfo, Capability, Device, Error, ErrorKind, Iov, Key, Location, MemoryRegion, NodeId,
    ObjectInfo, ObjectSpec, OpGet, OpPublish, OpPut, Placement, Replica, Result, Retention,
    RpGet, RpPut, SlotInfo, Tier,
};

use crate::cuda::{self, Gpus};
use crate::pinned::{PinnedBuf, PinnedPool};
use crate::topology::Topology;

pub struct LocalBuilder {
    node: NodeId,
    dram: Vec<(u16, usize)>,
    gpus: Option<Vec<u16>>,
}

impl Default for LocalBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalBuilder {
    pub fn new() -> Self {
        Self {
            node: NodeId(0),
            dram: Vec::new(),
            gpus: None,
        }
    }

    pub fn node(mut self, node: NodeId) -> Self {
        self.node = node;
        self
    }

    /// Reserve a pinned DRAM pool of `bytes` on NUMA node `numa`.
    pub fn dram(mut self, numa: u16, bytes: usize) -> Self {
        self.dram.push((numa, bytes));
        self
    }

    /// Restrict to these GPU indices (default: all visible).
    pub fn gpus(mut self, gpus: Vec<u16>) -> Self {
        self.gpus = Some(gpus);
        self
    }

    pub fn build(self) -> Result<Local> {
        cuda::init()?;
        let indices = match self.gpus {
            Some(g) => g,
            None => (0..cuda::device_count()?).collect(),
        };
        let gpus = Gpus::open(&indices)?;
        let topo = Topology::detect(
            &gpus
                .devices
                .iter()
                .map(|g| (g.index, g.pci_bus_id.clone()))
                .collect::<Vec<_>>(),
        );
        // Registration needs a current context.
        if let Some(g) = gpus.devices.first() {
            g.ctx.bind_to_thread().ok();
        }
        let mut pools = HashMap::new();
        for (numa, bytes) in &self.dram {
            if !topo.numa_nodes.contains(numa) {
                return Err(Error::new(ErrorKind::InvalidInput, "unknown NUMA node")
                    .with_context("numa", numa));
            }
            let pool = PinnedPool::new(&topo, *numa, *bytes)?;
            pools.insert(*numa, pool);
        }
        let mut devices: Vec<Device> = pools.keys().map(|n| Device::cpu(*n)).collect();
        devices.sort();
        devices.extend(gpus.devices.iter().map(|g| Device::gpu(g.index)));
        let info = AccessInfo {
            name: "local",
            node: self.node,
            devices: devices.clone(),
            capability: Capability::all(),
        };
        let fifo = devices.iter().map(|d| (*d, VecDeque::new())).collect();
        Ok(Local {
            inner: Arc::new(Inner {
                info: Arc::new(info),
                topo,
                gpus,
                pools,
                state: Mutex::new(State {
                    objects: BTreeMap::new(),
                    fifo,
                }),
            }),
        })
    }
}

#[derive(Clone)]
pub struct Local {
    inner: Arc<Inner>,
}

impl Local {
    pub fn builder() -> LocalBuilder {
        LocalBuilder::new()
    }

    pub fn topology(&self) -> &Topology {
        &self.inner.topo
    }

    /// (numa, capacity, free) per pool.
    pub fn pool_stats(&self) -> Vec<(u16, usize, usize)> {
        let mut v: Vec<_> = self
            .inner
            .pools
            .values()
            .map(|p| (p.numa(), p.capacity(), p.free_bytes()))
            .collect();
        v.sort();
        v
    }
}

struct Inner {
    info: Arc<AccessInfo>,
    topo: Topology,
    gpus: Gpus,
    pools: HashMap<u16, Arc<PinnedPool>>,
    state: Mutex<State>,
}

struct State {
    objects: BTreeMap<Bytes, Object>,
    /// Per store-owned device: insertion order of replicas for eviction.
    fifo: HashMap<Device, VecDeque<(Bytes, u32)>>,
}

struct Object {
    spec: ObjectSpec,
    slots: Vec<Slot>,
}

#[derive(Default)]
struct Slot {
    replicas: Vec<ReplicaEntry>,
    writing: bool,
}

#[derive(Clone)]
struct ReplicaEntry {
    device: Device,
    tier: Tier,
    misplaced: bool,
    data: Arc<ReplicaData>,
}

enum ReplicaData {
    Pinned(PinnedBuf),
    External { segs: Vec<Seg>, region_id: u64 },
}

#[derive(Clone, Copy, Debug)]
struct Seg {
    addr: u64,
    len: u64,
}

impl ReplicaData {
    fn segs(&self) -> Vec<Seg> {
        match self {
            ReplicaData::Pinned(b) => vec![Seg {
                addr: b.as_ptr() as u64,
                len: b.len() as u64,
            }],
            ReplicaData::External { segs, .. } => segs.clone(),
        }
    }
}

struct RegionGuard {
    inner: Weak<Inner>,
    id: u64,
    host_registered: Option<*mut u8>,
}
// SAFETY: raw pointer is only used for unregistration.
unsafe impl Send for RegionGuard {}
unsafe impl Sync for RegionGuard {}

impl Drop for RegionGuard {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.retire_region(self.id);
        }
        if let Some(p) = self.host_registered {
            // SAFETY: registered by us in `register`.
            unsafe { cuda::host_unregister(p) };
        }
    }
}

fn iov_segs(iovs: &[Iov<'_>]) -> Vec<Seg> {
    iovs.iter()
        .map(|i| Seg {
            addr: i.as_ptr() as u64,
            len: i.len,
        })
        .collect()
}

/// Walk two segment lists in lockstep, starting `src_offset` into `src`,
/// copying `total` bytes through `f(src_addr, dst_addr, len)`.
fn zip_copy(
    src: &[Seg],
    mut src_offset: u64,
    dst: &[Seg],
    total: u64,
    mut f: impl FnMut(u64, u64, u64) -> Result<()>,
) -> Result<()> {
    let mut si = 0usize;
    while si < src.len() && src_offset >= src[si].len {
        src_offset -= src[si].len;
        si += 1;
    }
    let mut s_off = src_offset;
    let mut di = 0usize;
    let mut d_off = 0u64;
    let mut left = total;
    while left > 0 {
        if si >= src.len() || di >= dst.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "copy ran past a segment list"));
        }
        let n = (src[si].len - s_off).min(dst[di].len - d_off).min(left);
        f(src[si].addr + s_off, dst[di].addr + d_off, n)?;
        s_off += n;
        d_off += n;
        left -= n;
        if s_off == src[si].len {
            si += 1;
            s_off = 0;
        }
        if d_off == dst[di].len {
            di += 1;
            d_off = 0;
        }
    }
    Ok(())
}

impl Inner {
    fn location(&self, device: Device, tier: Tier) -> Location {
        Location {
            node: self.info.node,
            device,
            tier,
        }
    }

    fn retire_region(&self, region_id: u64) {
        let mut st = self.state.lock().unwrap();
        for obj in st.objects.values_mut() {
            for slot in &mut obj.slots {
                slot.replicas.retain(
                    |r| !matches!(&*r.data, ReplicaData::External { region_id: id, .. } if *id == region_id),
                );
            }
        }
    }

    /// Move `total` bytes between arbitrary devices using the right engine.
    fn copy(&self, src_dev: Device, src: &[Seg], src_offset: u64, dst_dev: Device, dst: &[Seg], total: u64) -> Result<()> {
        match (src_dev, dst_dev) {
            (Device::Cpu { .. }, Device::Cpu { .. }) => zip_copy(src, src_offset, dst, total, |s, d, n| {
                // SAFETY: both sides are registered host memory validated by the caller.
                unsafe { std::ptr::copy_nonoverlapping(s as *const u8, d as *mut u8, n as usize) };
                Ok(())
            }),
            (Device::Cpu { .. }, Device::Gpu { index }) => {
                let g = self.gpu(index)?;
                zip_copy(src, src_offset, dst, total, |s, d, n| unsafe {
                    // SAFETY: host src / device dst validated; synced below.
                    cuda::h2d_async(d, s as *const u8, n as usize, &g.stream)
                })?;
                cuda::sync(&g.stream)
            }
            (Device::Gpu { index }, Device::Cpu { .. }) => {
                let g = self.gpu(index)?;
                zip_copy(src, src_offset, dst, total, |s, d, n| unsafe {
                    // SAFETY: device src / host dst validated; synced below.
                    cuda::d2h_async(d as *mut u8, s, n as usize, &g.stream)
                })?;
                cuda::sync(&g.stream)
            }
            (Device::Gpu { index: si }, Device::Gpu { index: di }) => {
                let sg = self.gpu(si)?;
                let dg = self.gpu(di)?;
                zip_copy(src, src_offset, dst, total, |s, d, n| unsafe {
                    // SAFETY: both device allocations validated; synced below.
                    cuda::peer_async(d, &dg.ctx, s, &sg.ctx, n as usize, &dg.stream)
                })?;
                cuda::sync(&dg.stream)
            }
        }
    }

    fn gpu(&self, index: u16) -> Result<&crate::cuda::Gpu> {
        self.gpus
            .get(index)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "gpu not served").with_context("gpu", index))
    }

    /// Allocate `len` bytes on `numa`, evicting Cache replicas FIFO if needed.
    fn alloc_or_evict(&self, numa: u16, len: usize, protect: &Bytes) -> Result<PinnedBuf> {
        let pool = self
            .pools
            .get(&numa)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "no pool on device").with_context("device", Device::cpu(numa)))?;
        if let Some(b) = pool.alloc(len) {
            return Ok(b);
        }
        if len > pool.capacity() {
            return Err(no_space(numa, len));
        }
        let device = Device::cpu(numa);
        loop {
            let mut st = self.state.lock().unwrap();
            let mut deferred = Vec::new();
            let mut evicted_any = false;
            while let Some((key, idx)) = st.fifo.get_mut(&device).unwrap().pop_front() {
                if key == *protect {
                    deferred.push((key, idx));
                    continue;
                }
                let Some(obj) = st.objects.get_mut(&key) else { continue };
                if obj.spec.retention != Retention::Cache {
                    continue;
                }
                let slot = &mut obj.slots[idx as usize];
                let before = slot.replicas.len();
                slot.replicas
                    .retain(|r| !(r.device == device && matches!(&*r.data, ReplicaData::Pinned(_))));
                if slot.replicas.len() != before {
                    evicted_any = true;
                }
                if obj.slots.iter().all(|s| s.replicas.is_empty() && !s.writing) {
                    st.objects.remove(&key);
                }
                if evicted_any {
                    break;
                }
            }
            let q = st.fifo.get_mut(&device).unwrap();
            for e in deferred.into_iter().rev() {
                q.push_front(e);
            }
            drop(st);
            if let Some(b) = pool.alloc(len) {
                return Ok(b);
            }
            if !evicted_any {
                return Err(no_space(numa, len));
            }
        }
    }
}

fn no_space(numa: u16, len: usize) -> Error {
    Error::new(ErrorKind::NoSpace, "pool cannot fit slot")
        .with_context("device", Device::cpu(numa))
        .with_context("len", len)
}

fn tier_rank(t: Tier) -> u8 {
    match t {
        Tier::Dram => 0,
        Tier::External => 1,
        Tier::Ssd => 2,
    }
}

impl Access for Local {
    fn info(&self) -> Arc<AccessInfo> {
        self.inner.info.clone()
    }

    unsafe fn register(&self, ptr: *mut u8, len: u64, device: Device) -> Result<MemoryRegion> {
        let id = MemoryRegion::next_id();
        let host_registered = match device {
            Device::Gpu { index } => {
                self.inner.gpu(index)?;
                None
            }
            // SAFETY: caller promises the mapping is valid for the region's life.
            Device::Cpu { .. } => unsafe { cuda::host_register(ptr, len as usize) }.then_some(ptr),
        };
        let guard = RegionGuard {
            inner: Arc::downgrade(&self.inner),
            id,
            host_registered,
        };
        // SAFETY: forwarded contract.
        Ok(unsafe { MemoryRegion::with_id(id, ptr, len, device, Box::new(guard)) })
    }

    async fn put(&self, key: Key, op: OpPut<'_>) -> Result<RpPut> {
        op.validate()?;
        let inner = &*self.inner;
        let spec = op.spec();
        let kb = key.0.clone();

        // Phase 1: reserve key + claim writable slots.
        let writable: Vec<bool> = {
            let mut st = inner.state.lock().unwrap();
            let obj = match st.objects.get_mut(&kb) {
                Some(obj) => {
                    if obj.spec != spec {
                        return Err(Error::new(ErrorKind::SpecMismatch, "existing object has a different spec")
                            .with_context("key", format!("{key:?}")));
                    }
                    obj
                }
                None => st.objects.entry(kb.clone()).or_insert_with(|| Object {
                    spec: spec.clone(),
                    slots: (0..spec.slots.len()).map(|_| Slot::default()).collect(),
                }),
            };
            obj.slots
                .iter_mut()
                .map(|s| {
                    let w = s.replicas.is_empty() && !s.writing;
                    if w {
                        s.writing = true;
                    }
                    w
                })
                .collect()
        };

        // Phase 2: place + copy, one slot at a time.
        let mut results = Vec::with_capacity(op.slots.len());
        for (i, ps) in op.slots.iter().enumerate() {
            if !writable[i] {
                results.push(Err(Error::new(ErrorKind::AlreadyExists, "slot already written").with_context("slot", i)));
                continue;
            }
            let r = self.put_slot(inner, &kb, i, ps);
            if r.is_err() {
                let mut st = inner.state.lock().unwrap();
                if let Some(obj) = st.objects.get_mut(&kb) {
                    obj.slots[i].writing = false;
                }
            }
            results.push(r.map_err(|e| e.with_context("slot", i)));
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
        let spec = obj
            .spec
            .slots
            .get(idx)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "slot index out of range").with_context("slot", idx))?;
        if spec.len != total {
            return Err(Error::new(ErrorKind::InvalidInput, "published length differs from slot length")
                .with_context("len", total)
                .with_context("expected", spec.len));
        }
        obj.slots[idx].replicas.push(ReplicaEntry {
            device,
            tier: Tier::External,
            misplaced: false,
            data: Arc::new(ReplicaData::External {
                segs: iov_segs(op.src),
                region_id: op.src[0].region.id(),
            }),
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
        let (replica, slot_len) = {
            let st = inner.state.lock().unwrap();
            let obj = st
                .objects
                .get(&key.0)
                .ok_or_else(|| Error::new(ErrorKind::NotFound, "no such key").with_context("key", format!("{key:?}")))?;
            let idx = op.slot.0 as usize;
            let spec = obj
                .spec
                .slots
                .get(idx)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "slot index out of range").with_context("slot", idx))?;
            let replica = obj.slots[idx]
                .replicas
                .iter()
                .min_by_key(|r| (inner.topo.distance(dst_device, r.device), tier_rank(r.tier)))
                .cloned()
                .ok_or_else(|| {
                    Error::new(ErrorKind::Evicted, "slot has no replica")
                        .with_context("key", format!("{key:?}"))
                        .with_context("slot", idx)
                })?;
            (replica, spec.len)
        };
        if op.src_offset.checked_add(dst_len).is_none_or(|e| e > slot_len) {
            return Err(Error::new(ErrorKind::InvalidInput, "read range exceeds slot")
                .with_context("src_offset", op.src_offset)
                .with_context("len", dst_len)
                .with_context("slot_len", slot_len));
        }
        // Lock released; `replica.data` keeps the bytes alive through the copy.
        inner.copy(
            replica.device,
            &replica.data.segs(),
            op.src_offset,
            dst_device,
            &iov_segs(op.dst),
            dst_len,
        )?;
        Ok(RpGet {
            from: inner.location(replica.device, replica.tier),
        })
    }

    async fn remove(&self, keys: &[Key]) -> Result<()> {
        let mut st = self.inner.state.lock().unwrap();
        for k in keys {
            st.objects.remove(&k.0);
        }
        Ok(())
    }

    async fn remove_prefix(&self, prefix: &[u8]) -> Result<u64> {
        let mut st = self.inner.state.lock().unwrap();
        let keys: Vec<Bytes> = st
            .objects
            .range(Bytes::copy_from_slice(prefix)..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, _)| k.clone())
            .collect();
        for k in &keys {
            st.objects.remove(k);
        }
        Ok(keys.len() as u64)
    }
}

impl Local {
    fn put_slot(&self, inner: &Inner, kb: &Bytes, i: usize, ps: &pegastore::PutSlot<'_>) -> Result<()> {
        let len = ps.spec.len;
        let src_dev = ps.src[0].device();
        let src_segs = iov_segs(ps.src);

        let numa_of = |d: &Device| -> Result<u16> {
            match d {
                Device::Cpu { numa } if inner.pools.contains_key(numa) => Ok(*numa),
                Device::Cpu { .. } => Err(Error::new(ErrorKind::InvalidInput, "no pool on device").with_context("device", *d)),
                Device::Gpu { .. } => Err(Error::new(ErrorKind::InvalidInput, "GPU memory is not a store tier; use publish")
                    .with_context("device", *d)),
            }
        };
        let mut pool_order: Vec<u16> = inner.pools.keys().copied().collect();
        pool_order.sort_unstable();

        // Resolve placement into (numa, misplaced) targets.
        let targets: Vec<(u16, bool)> = match &ps.spec.placement {
            Placement::Strict(d) => vec![(numa_of(d)?, false)],
            Placement::Each(ds) => ds.iter().map(|d| numa_of(d).map(|n| (n, false))).collect::<Result<_>>()?,
            Placement::Prefer(d) => {
                let n = numa_of(d)?;
                std::iter::once((n, false))
                    .chain(pool_order.iter().filter(|x| **x != n).map(|x| (*x, true)))
                    .collect()
            }
            Placement::Anywhere => pool_order.iter().map(|n| (*n, false)).collect(),
        };
        let each = matches!(ps.spec.placement, Placement::Each(_));

        let mut placed: Vec<(u16, bool, PinnedBuf)> = Vec::new();
        let mut last_err = None;
        for (numa, misplaced) in targets {
            match inner.alloc_or_evict(numa, len as usize, kb) {
                Ok(buf) => {
                    placed.push((numa, misplaced, buf));
                    if !each {
                        break;
                    }
                }
                Err(e) => {
                    last_err = Some(e);
                    if each {
                        break;
                    }
                }
            }
        }
        if placed.is_empty() || (each && last_err.is_some()) {
            return Err(last_err.unwrap_or_else(|| Error::new(ErrorKind::NoSpace, "no device available")));
        }

        // Copy into every target (D2H per target: the copy engine is faster
        // than a host-side memcpy between sockets).
        for (numa, _, buf) in &placed {
            let dst = [Seg {
                addr: buf.as_ptr() as u64,
                len,
            }];
            inner.copy(src_dev, &src_segs, 0, Device::cpu(*numa), &dst, len)?;
        }

        // Phase 3: publish replicas.
        let mut st = inner.state.lock().unwrap();
        let obj = st
            .objects
            .get_mut(kb)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "object removed during put"))?;
        let slot = &mut obj.slots[i];
        let mut charged = Vec::with_capacity(placed.len());
        for (numa, misplaced, buf) in placed {
            slot.replicas.push(ReplicaEntry {
                device: Device::cpu(numa),
                tier: Tier::Dram,
                misplaced,
                data: Arc::new(ReplicaData::Pinned(buf)),
            });
            charged.push(Device::cpu(numa));
        }
        slot.writing = false;
        for d in charged {
            st.fifo.get_mut(&d).unwrap().push_back((kb.clone(), i as u32));
        }
        Ok(())
    }
}
