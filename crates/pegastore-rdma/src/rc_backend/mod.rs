mod runtime;
mod session;
mod state;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use log::{debug, info, warn};
use parking_lot::Mutex;
use sideway::ibverbs::AccessFlags;
use sideway::ibverbs::memory_region::MemoryRegion;
use sideway::ibverbs::protection_domain::ProtectionDomain;

use self::runtime::RcRuntime;
use self::session::{RcSession, RdmaOp};
use self::state::{AddrConnection, ConnNic, RcBackendState, RegisteredMemoryEntry};
use std::os::fd::RawFd;

use mea::oneshot;

use crate::engine::{NicHandshake, RegionDescriptor, TransferDesc, TransferOp};
use crate::error::{Result, TransferError};
use crate::numa::NumaNode;

fn mr_access() -> AccessFlags {
    AccessFlags::LocalWrite | AccessFlags::RemoteWrite | AccessFlags::RemoteRead
}

struct NicGroup {
    nic_indices: Vec<usize>,
    rr_counter: AtomicUsize,
}

impl NicGroup {
    fn next(&self) -> usize {
        let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed);
        self.nic_indices[idx % self.nic_indices.len()]
    }
}

struct NumaRoundRobin {
    /// NUMA node → NIC group on that node.
    groups: HashMap<NumaNode, NicGroup>,
    /// Fallback for unknown NUMA or unmatched nodes (all NICs).
    fallback: NicGroup,
}

impl NumaRoundRobin {
    fn from_runtimes(runtimes: &[Arc<RcRuntime>]) -> Self {
        let all_indices: Vec<usize> = (0..runtimes.len()).collect();

        // Group NIC indices by NUMA node.
        let mut map: HashMap<NumaNode, Vec<usize>> = HashMap::new();
        for (i, rt) in runtimes.iter().enumerate() {
            if rt.numa_node.is_valid() {
                map.entry(rt.numa_node).or_default().push(i);
            }
        }


        let groups: HashMap<NumaNode, NicGroup> = map
            .into_iter()
            .map(|(node, indices)| {
                (
                    node,
                    NicGroup {
                        nic_indices: indices,
                        rr_counter: AtomicUsize::new(0),
                    },
                )
            })
            .collect();

        let fallback = NicGroup {
            nic_indices: all_indices,
            rr_counter: AtomicUsize::new(0),
        };

        Self {
            groups,
            fallback,
        }
    }

    fn pick(&self, numa: NumaNode) -> usize {
        if numa.is_valid()
            && let Some(group) = self.groups.get(&numa)
        {
            return group.next();
        }
        self.fallback.next()
    }
}

pub(crate) enum GetOrPrepareResult {
    Existing,
    AlreadyConnecting,
    NeedHandshake(Vec<NicHandshake>),
}

pub(crate) struct RcBackend {
    runtimes: Vec<Arc<RcRuntime>>,
    state: Arc<Mutex<RcBackendState>>,
    psn_counter: AtomicU64,
    numa_rr: NumaRoundRobin,
    qps_per_peer: usize,
}

impl RcBackend {
    pub(crate) fn new(nic_names: &[String], qps_per_peer: usize) -> Result<Self> {
        crate::init_logging();
        if nic_names.is_empty() {
            return Err(TransferError::InvalidArgument("nic_names is empty"));
        }
        if qps_per_peer == 0 {
            return Err(TransferError::InvalidArgument("qps_per_peer must be > 0"));
        }
        let mut runtimes = Vec::with_capacity(nic_names.len());
        for name in nic_names {
            if name.trim().is_empty() {
                return Err(TransferError::InvalidArgument("nic_name is empty"));
            }
            let runtime = RcRuntime::open(name)?;
            runtimes.push(runtime);
        }
        let nic_count = runtimes.len();
        let numa_rr = NumaRoundRobin::from_runtimes(&runtimes);
        info!(
            "RC backend init: nics={}, qps_per_peer={}",
            nic_count, qps_per_peer
        );
        Ok(Self {
            runtimes,
            state: Arc::new(Mutex::new(RcBackendState::new(nic_count))),
            psn_counter: AtomicU64::new(1),
            numa_rr,
            qps_per_peer,
        })
    }

    fn nic_count(&self) -> usize {
        self.runtimes.len()
    }

    pub(crate) fn num_nics(&self) -> usize {
        self.runtimes.len()
    }

    /// # Safety
    /// See [`crate::TransferEngine::register_host`].
    pub(crate) unsafe fn register_host(&self, addr: u64, len: usize) -> Result<RegionDescriptor> {
        // One move_pages at registration instead of one per op: pegastore's
        // pools are NUMA-homogeneous by construction.
        let numa = crate::numa::query_pages_numa(&[addr as *const u8])[0];
        self.register_with(addr, len, numa, "host", |pd| {
            // SAFETY: caller guarantees `[addr, addr + len)` is valid host memory.
            unsafe { pd.reg_mr(addr as usize, len, mr_access()) }.map_err(|e| e.to_string())
        })
    }

    /// # Safety
    /// See [`crate::TransferEngine::register_dmabuf`].
    pub(crate) unsafe fn register_dmabuf(
        &self,
        addr: u64,
        len: usize,
        fd: RawFd,
        offset: u64,
        numa: NumaNode,
    ) -> Result<RegionDescriptor> {
        self.register_with(addr, len, numa, "dmabuf", |pd| {
            // SAFETY: caller guarantees `fd` exports `[offset, offset + len)`;
            // `addr` becomes the IOVA so descriptors carry device VAs.
            unsafe { pd.reg_dmabuf_mr(offset, len, addr, fd, mr_access()) }
                .map_err(|e| e.to_string())
        })
    }

    fn register_with(
        &self,
        addr: u64,
        len: usize,
        numa: NumaNode,
        kind: &str,
        reg: impl Fn(&Arc<ProtectionDomain>) -> std::result::Result<Arc<MemoryRegion>, String>,
    ) -> Result<RegionDescriptor> {
        if len == 0 {
            return Err(TransferError::InvalidArgument("len must be non-zero"));
        }
        let mut mrs = Vec::with_capacity(self.nic_count());
        for runtime in &self.runtimes {
            mrs.push(reg(&runtime.pd).map_err(|e| {
                TransferError::Backend(format!(
                    "{kind} MR registration failed on nic={}: {e}",
                    runtime.nic_name
                ))
            })?);
        }
        let rkeys: Vec<u32> = mrs.iter().map(|mr| mr.rkey()).collect();
        let mut state = self.state.lock();
        Arc::make_mut(&mut state.registered).insert(RegisteredMemoryEntry {
            base_ptr: addr,
            len,
            numa,
            mrs,
        })?;
        debug!(
            "memory registered: kind={kind}, addr={addr:#x}, len={len}, numa={numa}, nics={}",
            self.nic_count()
        );
        Ok(RegionDescriptor {
            addr,
            len: len as u64,
            rkeys,
        })
    }

    pub(crate) fn unregister(&self, addr: u64) -> Result<()> {
        let mut state = self.state.lock();
        let removed = Arc::make_mut(&mut state.registered).remove(addr);
        if removed.is_none() {
            return Err(TransferError::MemoryNotRegistered { ptr: addr });
        }
        debug!("memory unregistered: addr={addr:#x}");
        Ok(())
    }

    /// Create N RC QPs per NIC in INIT state, push to per-NIC pending queues,
    /// return per-NIC handshake data (each NIC carries N endpoints).
    fn prepare_handshake(&self) -> Result<Vec<NicHandshake>> {
        let mut nic_handshakes = Vec::with_capacity(self.nic_count());

        // Create all NIC × N QPs before locking state.
        let mut sessions_per_nic: Vec<Vec<Arc<RcSession>>> =
            (0..self.nic_count()).map(|_| Vec::new()).collect();
        for (nic_idx, runtime) in self.runtimes.iter().enumerate() {
            for qp_idx in 0..self.qps_per_peer {
                let psn_seed = self.psn_counter.fetch_add(1, Ordering::Relaxed);
                let session = RcSession::create(runtime, psn_seed).map_err(|e| {
                    TransferError::Backend(format!(
                        "QP creation failed on nic={} qp_idx={qp_idx}: {e}",
                        runtime.nic_name
                    ))
                })?;
                sessions_per_nic[nic_idx].push(session);
            }
        }

        let mut state = self.state.lock();
        for (nic_idx, sessions) in sessions_per_nic.into_iter().enumerate() {
            let endpoints: Vec<_> = sessions.iter().map(|s| s.local_endpoint).collect();
            for s in sessions {
                state.nics[nic_idx].pending.push_back(s);
            }
            nic_handshakes.push(NicHandshake { endpoints });
        }

        Ok(nic_handshakes)
    }

    /// Check if connected to remote_addr; if not, prepare local QPs.
    pub(crate) fn get_or_prepare(&self, remote_addr: &str) -> Result<GetOrPrepareResult> {
        {
            let mut state = self.state.lock();
            if state.addr_connections.contains_key(remote_addr) {
                return Ok(GetOrPrepareResult::Existing);
            }
            if state.connecting.contains(remote_addr) {
                return Ok(GetOrPrepareResult::AlreadyConnecting);
            }
            state.connecting.insert(remote_addr.to_string());
        }
        match self.prepare_handshake() {
            Ok(nics) => Ok(GetOrPrepareResult::NeedHandshake(nics)),
            Err(e) => {
                let removed = self.state.lock().connecting.remove(remote_addr);
                debug_assert!(removed, "connecting set should contain {remote_addr}");
                Err(e)
            }
        }
    }

    /// Complete a connection after handshake exchange.
    pub(crate) fn complete_handshake_for(
        &self,
        remote_addr: &str,
        local_nics: Vec<NicHandshake>,
        remote_nics: &[NicHandshake],
    ) -> Result<()> {
        let nic_count = self.nic_count();
        if remote_nics.len() != nic_count {
            return Err(TransferError::InvalidArgument(
                "remote NIC count mismatch in handshake",
            ));
        }
        for (nic_idx, nic) in remote_nics.iter().enumerate() {
            if nic.endpoints.len() != self.qps_per_peer {
                return Err(TransferError::Backend(format!(
                    "remote qps_per_peer mismatch on nic={nic_idx}: local={}, remote={}",
                    self.qps_per_peer,
                    nic.endpoints.len()
                )));
            }
        }

        // Pop pending sessions by matching QPN from local_nics (not blind FIFO).
        // Concurrent prepare/complete for different remote addrs could reorder the
        // queue, so we must find our own sessions by QPN.
        let pending: Vec<Vec<Arc<RcSession>>> = {
            let mut state = self.state.lock();
            // If already connected (concurrent request beat us), remove our pending sessions
            if state.addr_connections.contains_key(remote_addr) {
                for (nic_idx, nic) in local_nics.iter().enumerate() {
                    for ep in &nic.endpoints {
                        state.nics[nic_idx].remove_pending_by_qpn(ep.qp_num);
                    }
                }
                state.connecting.remove(remote_addr);
                info!("handshake won by concurrent path: remote={remote_addr}");
                return Ok(());
            }
            let mut sessions_per_nic: Vec<Vec<Arc<RcSession>>> = (0..nic_count)
                .map(|_| Vec::with_capacity(self.qps_per_peer))
                .collect();
            for (nic_idx, nic) in local_nics.iter().enumerate() {
                for ep in &nic.endpoints {
                    let session = state.nics[nic_idx].remove_pending_by_qpn(ep.qp_num).ok_or(
                        TransferError::Backend("no pending session to complete".to_string()),
                    )?;
                    sessions_per_nic[nic_idx].push(session);
                }
            }
            sessions_per_nic
        };

        // Connect outside lock — pair local QP i with remote QP i in handshake order.
        for (nic_idx, sessions) in pending.iter().enumerate() {
            let remote_eps = &remote_nics[nic_idx].endpoints;
            for (qp_idx, session) in sessions.iter().enumerate() {
                session.connect(&self.runtimes[nic_idx], &remote_eps[qp_idx])?;
            }
        }

        // Validate snapshots and assemble the connection before locking.
        let conn_nics: Vec<ConnNic> = pending
            .into_iter()
            .map(|sessions| ConnNic {
                sessions: Arc::new(sessions),
                rr_counter: AtomicUsize::new(0),
            })
            .collect();

        let mut state = self.state.lock();
        let removed = state.connecting.remove(remote_addr);
        debug_assert!(removed, "connecting set should contain {remote_addr}");
        let local_qpns: Vec<Vec<u32>> = local_nics
            .iter()
            .map(|n| n.endpoints.iter().map(|e| e.qp_num).collect())
            .collect();
        let remote_qpns: Vec<Vec<u32>> = remote_nics
            .iter()
            .map(|n| n.endpoints.iter().map(|e| e.qp_num).collect())
            .collect();
        info!(
            "RDMA connection established: remote={remote_addr}, qps_per_peer={}, local_qpns={local_qpns:?}, remote_qpns={remote_qpns:?}",
            self.qps_per_peer
        );
        state.addr_connections.insert(
            remote_addr.to_string(),
            AddrConnection {
                nics: conn_nics,
                local_nics,
            },
        );
        Ok(())
    }

    /// Drop pending sessions created by get_or_prepare when handshake failed.
    pub(crate) fn abort_handshake(&self, remote_addr: &str, local_nics: &[NicHandshake]) {
        let mut state = self.state.lock();
        let removed = state.connecting.remove(remote_addr);
        debug_assert!(removed, "connecting set should contain {remote_addr}");
        for (nic_idx, nic) in local_nics.iter().enumerate() {
            for ep in &nic.endpoints {
                state.nics[nic_idx].remove_pending_by_qpn(ep.qp_num);
            }
        }
        warn!("handshake aborted: remote={remote_addr}");
    }

    /// Get local NicHandshake metadata for an established connection.
    pub(crate) fn local_meta_for_addr(&self, remote_addr: &str) -> Option<Vec<NicHandshake>> {
        let state = self.state.lock();
        state
            .addr_connections
            .get(remote_addr)
            .map(|c| c.local_nics.clone())
    }

    /// Remove connection state on transfer failure. The connection owns its
    /// sessions, so in-flight work keeps its QPs alive through their Arcs.
    pub(crate) fn invalidate_connection(&self, remote_addr: &str) {
        let mut state = self.state.lock();
        if state.addr_connections.remove(remote_addr).is_some() {
            info!("connection invalidated: remote={remote_addr}");
        }
    }

    pub(crate) fn num_qps(&self) -> usize {
        self.state.lock().num_qps()
    }

    /// One receiver per NIC that had work; each yields bytes transferred on that NIC.
    pub(crate) fn batch_transfer_async(
        &self,
        op: TransferOp,
        remote_addr: &str,
        descs: &[TransferDesc<'_>],
    ) -> Result<Vec<oneshot::Receiver<Result<usize>>>> {
        if descs.is_empty() {
            return Ok(Vec::new());
        }
        let nic_count = self.nic_count();
        let lookup_start = Instant::now();

        // Snapshot the registration map and the connection's sessions under
        // the lock; resolution and posting run outside it.
        let (registered, conn_nics) = {
            let state = self.state.lock();
            let conn = state
                .addr_connections
                .get(remote_addr)
                .ok_or(TransferError::Backend(format!(
                    "no connection for remote addr {remote_addr}"
                )))?;
            let nics: Vec<(Arc<Vec<Arc<RcSession>>>, usize)> = conn
                .nics
                .iter()
                .map(|nic| {
                    (
                        Arc::clone(&nic.sessions),
                        nic.rr_counter.fetch_add(1, Ordering::Relaxed),
                    )
                })
                .collect();
            (Arc::clone(&state.registered), nics)
        };

        // Resolve each op: local MR by registered range, NIC by that
        // region's NUMA node, remote rkey from the peer's descriptor (NICs
        // are index-paired across the two engines).
        let mut per_nic: Vec<Vec<RdmaOp>> = (0..nic_count).map(|_| Vec::new()).collect();
        for desc in descs {
            if desc.len == 0 {
                return Err(TransferError::InvalidArgument("len must be non-zero"));
            }
            let entry = registered
                .find_entry(desc.local, desc.len)
                .ok_or(TransferError::MemoryNotRegistered { ptr: desc.local })?;
            if !desc.region.contains(desc.remote, desc.len) {
                return Err(TransferError::InvalidArgument(
                    "remote range not covered by its region descriptor",
                ));
            }
            let nic_idx = self.numa_rr.pick(entry.numa);
            let remote_rkey = *desc.region.rkeys.get(nic_idx).ok_or(
                TransferError::InvalidArgument("region descriptor has no rkey for paired NIC"),
            )?;
            per_nic[nic_idx].push(RdmaOp {
                local_mr: Arc::clone(&entry.mrs[nic_idx]),
                local_ptr: desc.local,
                remote_ptr: desc.remote,
                len: desc.len,
                remote_rkey,
            });
        }

        // Spread each NIC's ops over its N sessions, rotating the start.
        let mut nic_work: Vec<(Arc<RcSession>, Vec<RdmaOp>)> = Vec::new();
        for (nic_idx, ops) in per_nic.into_iter().enumerate() {
            if ops.is_empty() {
                continue;
            }
            let (sessions, rot) = &conn_nics[nic_idx];
            let n = sessions.len();
            let mut buckets: Vec<Vec<RdmaOp>> =
                (0..n).map(|_| Vec::with_capacity(ops.len().div_ceil(n))).collect();
            for (i, rdma_op) in ops.into_iter().enumerate() {
                buckets[rot.wrapping_add(i) % n].push(rdma_op);
            }
            for (q_idx, prepared) in buckets.into_iter().enumerate() {
                if !prepared.is_empty() {
                    nic_work.push((Arc::clone(&sessions[q_idx]), prepared));
                }
            }
        }

        debug!(
            "batch_transfer_async_{:?}: nics_active={}/{}, chunks={}, lookup_ms={:.3}",
            op,
            nic_work.len(),
            nic_count,
            descs.len(),
            lookup_start.elapsed().as_secs_f64() * 1000.0,
        );

        let mut receivers = Vec::with_capacity(nic_work.len());
        for (session, prepared) in nic_work {
            receivers.push(session.transfer_batch_async(prepared, op)?);
        }
        Ok(receivers)
    }
}
