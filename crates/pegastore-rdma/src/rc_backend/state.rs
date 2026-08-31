use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use sideway::ibverbs::memory_region::MemoryRegion;

use super::session::RcSession;
use crate::engine::NicHandshake;
use crate::numa::NumaNode;
use crate::error::{Result, TransferError};

#[derive(Clone)]
pub(super) struct RegisteredMemoryEntry {
    pub(super) base_ptr: u64,
    pub(super) len: usize,
    /// NUMA node of the memory; drives NIC selection (UNKNOWN → any NIC).
    pub(super) numa: NumaNode,
    /// One MR per NIC (different PDs → different rkeys).
    pub(super) mrs: Vec<Arc<MemoryRegion>>,
}

/// Local registered memory, ordered by base pointer. Insertion rejects
/// overlapping regions so `find_entry` can resolve any address with a single
/// predecessor lookup. Shared as an RCU-style snapshot: register/unregister
/// clone-and-swap under the state lock, the transfer hot path reads its own
/// `Arc` outside the lock.
#[derive(Clone, Default)]
pub(super) struct LocalMemoryMap {
    entries: BTreeMap<u64, RegisteredMemoryEntry>,
}

impl LocalMemoryMap {
    /// Insert a region. Rejects any overlap with an existing registration,
    /// including an exact same-base duplicate: replacing would dereg the old
    /// MR whose rkey peers already hold from a handshake, so their next READ
    /// would fail with a remote access error — better to fail loudly here.
    pub(super) fn insert(&mut self, entry: RegisteredMemoryEntry) -> Result<()> {
        if entry.len == 0 {
            return Err(TransferError::InvalidArgument("len must be non-zero"));
        }
        let end =
            entry
                .base_ptr
                .checked_add(entry.len as u64)
                .ok_or(TransferError::InvalidArgument(
                    "memory region overflows address space",
                ))?;
        // Prior entries passed this same check, so their end cannot overflow.
        let overlap = self
            .entries
            .range(..=entry.base_ptr)
            .next_back()
            .filter(|(_, prev)| entry.base_ptr < prev.base_ptr + prev.len as u64)
            .or_else(|| {
                self.entries
                    .range(entry.base_ptr..)
                    .next()
                    .filter(|&(&next_base, _)| next_base < end)
            });
        if let Some((_, existing)) = overlap {
            return Err(TransferError::Backend(format!(
                "memory region [{:#x}, len={:#x}) overlaps registered region [{:#x}, len={:#x})",
                entry.base_ptr, entry.len, existing.base_ptr, existing.len
            )));
        }
        self.entries.insert(entry.base_ptr, entry);
        Ok(())
    }

    pub(super) fn remove(&mut self, base_ptr: u64) -> Option<RegisteredMemoryEntry> {
        self.entries.remove(&base_ptr)
    }



    /// Non-overlap makes the predecessor the only possible covering region.
    pub(super) fn find_entry(&self, ptr: u64, len: usize) -> Option<&RegisteredMemoryEntry> {
        let end = ptr.checked_add(len as u64)?;
        let (_, entry) = self.entries.range(..=ptr).next_back()?;
        (end <= entry.base_ptr + entry.len as u64).then_some(entry)
    }
}

/// Per-NIC state: sessions awaiting handshake completion. Established
/// sessions live inside their `AddrConnection`.
#[derive(Default)]
pub(super) struct PerNicState {
    /// Pre-connect sessions in FIFO order (first prepared, first connected).
    pub(super) pending: VecDeque<Arc<RcSession>>,
}

impl PerNicState {
    /// Remove a pending session by its local QPN. Returns the session if found.
    pub(super) fn remove_pending_by_qpn(&mut self, qpn: u32) -> Option<Arc<RcSession>> {
        let pos = self
            .pending
            .iter()
            .position(|s| s.local_endpoint.qp_num == qpn)?;
        self.pending.remove(pos)
    }
}

/// One NIC's slice of an established connection: the N connected sessions.
pub(super) struct ConnNic {
    pub(super) sessions: Arc<Vec<Arc<RcSession>>>,
    /// Round-robin counter for picking among the N sessions.
    pub(super) rr_counter: AtomicUsize,
}

/// The connection owns its sessions and snapshots directly. Keying them in a
/// shared map by remote QPN is unsound: QPNs are only unique within one
/// remote HCA, so two peers can collide, and the second handshake would
/// silently overwrite the first peer's sessions (and invalidating one peer
/// would destroy the other's).
pub(super) struct AddrConnection {
    /// Indexed by nic_idx.
    pub(super) nics: Vec<ConnNic>,
    pub(super) local_nics: Vec<NicHandshake>,
}

pub(super) struct RcBackendState {
    pub(super) registered: Arc<LocalMemoryMap>,
    pub(super) nics: Vec<PerNicState>,
    /// addr -> established connection info
    pub(super) addr_connections: HashMap<String, AddrConnection>,
    /// Addresses with a handshake in progress. Prevents concurrent
    /// `get_or_prepare` calls from creating duplicate QPs for the same peer.
    pub(super) connecting: HashSet<String>,
}

impl RcBackendState {
    pub(super) fn num_qps(&self) -> usize {
        self.addr_connections
            .values()
            .flat_map(|conn| conn.nics.iter())
            .map(|nic| nic.sessions.len())
            .sum()
    }

    pub(super) fn new(nic_count: usize) -> Self {
        Self {
            registered: Arc::new(LocalMemoryMap::default()),
            nics: (0..nic_count).map(|_| PerNicState::default()).collect(),
            addr_connections: HashMap::new(),
            connecting: HashSet::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(base_ptr: u64, len: usize) -> RegisteredMemoryEntry {
        RegisteredMemoryEntry {
            base_ptr,
            len,
            numa: NumaNode::UNKNOWN,
            mrs: Vec::new(),
        }
    }

    #[test]
    fn local_map_rejects_overlap_and_finds_covering_region() {
        let mut map = LocalMemoryMap::default();
        map.insert(entry(0x1000, 0x100)).expect("first region");
        map.insert(entry(0x3000, 0x100)).expect("disjoint region");

        // Overlap with predecessor and successor both rejected.
        assert!(map.insert(entry(0x10ff, 0x10)).is_err());
        assert!(map.insert(entry(0x2f80, 0x100)).is_err());
        // Same-base duplicate rejected (replace would dereg a broadcast MR).
        assert!(map.insert(entry(0x1000, 0x80)).is_err());
        // Adjacent regions (prev.end == base) allowed — K/V segments abut.
        map.insert(entry(0x1100, 0x100)).expect("adjacent region");
        // Zero-length rejected.
        assert!(map.insert(entry(0x5000, 0)).is_err());

        // Fully covered → hit; straddling a region boundary stays within that
        // region's entry → miss even though the next region abuts.
        assert_eq!(
            map.find_entry(0x1080, 0x20).map(|e| e.base_ptr),
            Some(0x1000)
        );
        assert!(map.find_entry(0x10f0, 0x20).is_none());
        assert!(map.find_entry(0x0f00, 0x10).is_none());
    }
}
