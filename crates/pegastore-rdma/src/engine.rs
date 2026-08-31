// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::os::fd::RawFd;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TransferError};
use crate::numa::NumaNode;
use crate::rc_backend::{GetOrPrepareResult, RcBackend};

/// RDMA operation type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOp {
    Read,
    Write,
}

/// What registering memory produces: the address range plus one rkey per
/// NIC of the registering engine (indexed like that engine's NIC list).
///
/// This is the only thing a peer needs to address the memory. It is
/// serializable so the owner can publish it out of band — in pegastore it
/// rides in replica metadata — and it is independent of any connection:
/// memory can be registered before, after, or without a handshake.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionDescriptor {
    pub addr: u64,
    pub len: u64,
    pub rkeys: Vec<u32>,
}

impl RegionDescriptor {
    /// True when `[ptr, ptr + len)` lies inside this region.
    pub fn contains(&self, ptr: u64, len: usize) -> bool {
        let Some(end) = ptr.checked_add(len as u64) else {
            return false;
        };
        ptr >= self.addr && end <= self.addr.saturating_add(self.len)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("region descriptor serialization should not fail")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (desc, consumed) = bincode::serde::decode_from_slice::<Self, _>(
            bytes,
            bincode::config::standard(),
        )
        .map_err(|_| TransferError::InvalidArgument("invalid region descriptor"))?;
        if consumed != bytes.len() {
            return Err(TransferError::InvalidArgument(
                "trailing bytes in region descriptor",
            ));
        }
        Ok(desc)
    }
}

/// A single RDMA transfer: `len` bytes between local address `local` and
/// remote address `remote`, the latter addressed through `region` (the
/// remote owner's [`RegionDescriptor`]).
#[derive(Clone, Copy, Debug)]
pub struct TransferDesc<'a> {
    pub local: u64,
    pub remote: u64,
    pub len: usize,
    pub region: &'a RegionDescriptor,
}

/// RC queue pair endpoint info exchanged during handshake.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RcEndpoint {
    pub(crate) gid: [u8; 16],
    pub(crate) lid: u16,
    pub(crate) qp_num: u32,
    pub(crate) psn: u32,
}

/// Per-NIC handshake data: N endpoints (one per QP).
///
/// The same NIC pair maintains N RC QPs so the caller can spread load across
/// them. Both peers must agree on N (set via `qps_per_peer` at backend init).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct NicHandshake {
    pub(crate) endpoints: Vec<RcEndpoint>,
}

/// Opaque handshake metadata exchanged between peers out of band.
///
/// Contains one [`NicHandshake`] per NIC. NICs are 1:1 mapped by index
/// between two machines (mlx5_0↔mlx5_0, etc.). Memory is *not* part of the
/// handshake; see [`RegionDescriptor`].
#[derive(Clone, Debug)]
pub struct HandshakeMetadata {
    pub(crate) nics: Vec<NicHandshake>,
}

#[derive(Serialize, Deserialize)]
struct WireHandshakeMetadata {
    nics: Vec<NicHandshake>,
}

impl HandshakeMetadata {
    pub fn to_bytes(&self) -> Vec<u8> {
        let wire = WireHandshakeMetadata {
            nics: self.nics.clone(),
        };
        bincode::serde::encode_to_vec(&wire, bincode::config::standard())
            .expect("handshake metadata serialization should not fail")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (wire, consumed) = bincode::serde::decode_from_slice::<WireHandshakeMetadata, _>(
            bytes,
            bincode::config::standard(),
        )
        .map_err(|_| TransferError::InvalidArgument("invalid handshake metadata"))?;
        if consumed != bytes.len() {
            return Err(TransferError::InvalidArgument(
                "trailing bytes in handshake metadata",
            ));
        }
        Ok(Self { nics: wire.nics })
    }
}

/// Connection status for a remote peer.
pub enum ConnectionStatus {
    /// Already connected; call batch_transfer_async directly.
    Existing,
    /// A handshake to this peer is already in progress.
    Connecting,
    /// Not connected. Exchange this local metadata with remote peer out of
    /// band, then call complete_handshake. On failure, call abort_handshake.
    Prepared(HandshakeMetadata),
}

pub struct TransferEngine {
    backend: RcBackend,
}

impl TransferEngine {
    pub fn new(nic_names: &[String], qps_per_peer: usize) -> Result<Self> {
        Ok(Self {
            backend: RcBackend::new(nic_names, qps_per_peer)?,
        })
    }

    /// Register host memory on every NIC. Its NUMA node (queried once from
    /// the first page) decides which NICs carry transfers touching it.
    ///
    /// # Safety
    /// `[addr, addr + len)` must be valid, mapped host memory that outlives
    /// the registration.
    pub unsafe fn register_host(&self, addr: u64, len: usize) -> Result<RegionDescriptor> {
        // SAFETY: forwarded contract.
        unsafe { self.backend.register_host(addr, len) }
    }

    /// Register device memory exported as a dma-buf (e.g. CUDA
    /// `cuMemGetHandleForAddressRange`). `addr` is the device virtual address
    /// used in transfer descriptors; `offset` is the range's offset inside
    /// the dma-buf. `numa` is the socket the device hangs off, used for NIC
    /// selection; pass `NumaNode::UNKNOWN` to spread across all NICs.
    ///
    /// # Safety
    /// `fd` must export at least `[offset, offset + len)`, and the mapping
    /// backing `addr` must stay alive for the registration's lifetime.
    pub unsafe fn register_dmabuf(
        &self,
        addr: u64,
        len: usize,
        fd: RawFd,
        offset: u64,
        numa: NumaNode,
    ) -> Result<RegionDescriptor> {
        // SAFETY: forwarded contract.
        unsafe { self.backend.register_dmabuf(addr, len, fd, offset, numa) }
    }

    /// Drop a registration by its base address. Peers still holding the
    /// descriptor will fail with a remote access error.
    pub fn unregister(&self, addr: u64) -> Result<()> {
        self.backend.unregister(addr)
    }

    /// Check if already connected to the remote peer; if not, prepare local
    /// QPs and return the handshake metadata that must be exchanged out of band.
    pub fn get_or_prepare(&self, remote_addr: &str) -> Result<ConnectionStatus> {
        match self.backend.get_or_prepare(remote_addr)? {
            GetOrPrepareResult::Existing => Ok(ConnectionStatus::Existing),
            GetOrPrepareResult::AlreadyConnecting => Ok(ConnectionStatus::Connecting),
            GetOrPrepareResult::NeedHandshake(nics) => {
                Ok(ConnectionStatus::Prepared(HandshakeMetadata { nics }))
            }
        }
    }

    /// Complete a connection after exchanging handshake metadata with the peer.
    pub fn complete_handshake(
        &self,
        remote_addr: &str,
        local_meta: &HandshakeMetadata,
        remote_meta: &HandshakeMetadata,
    ) -> Result<()> {
        self.backend
            .complete_handshake_for(remote_addr, local_meta.nics.clone(), &remote_meta.nics)
    }

    /// Drop pending sessions created by get_or_prepare when handshake failed.
    pub fn abort_handshake(&self, remote_addr: &str, local_meta: &HandshakeMetadata) {
        self.backend.abort_handshake(remote_addr, &local_meta.nics);
    }

    /// Return cached local handshake metadata for an established connection.
    pub fn local_meta_for(&self, remote_addr: &str) -> Option<HandshakeMetadata> {
        self.backend
            .local_meta_for_addr(remote_addr)
            .map(|nics| HandshakeMetadata { nics })
    }

    /// Remove cached connection state on transfer failure.
    pub fn invalidate_connection(&self, remote_addr: &str) {
        self.backend.invalidate_connection(remote_addr);
    }

    /// Submit a batch of RDMA READ or WRITE operations against a connected peer.
    ///
    /// Each op goes out on a NIC local to its registered memory's NUMA node
    /// (round-robin within the node). Returns one receiver per active NIC;
    /// each yields the bytes transferred on that NIC.
    ///
    /// The connection must be established via `get_or_prepare` +
    /// `complete_handshake` before calling this method. Every op's `local`
    /// range must be registered here and its `remote` range covered by
    /// `region`.
    pub fn batch_transfer_async(
        &self,
        op: TransferOp,
        remote_addr: &str,
        descs: &[TransferDesc<'_>],
    ) -> Result<Vec<mea::oneshot::Receiver<Result<usize>>>> {
        self.backend.batch_transfer_async(op, remote_addr, descs)
    }

    /// Number of active RC queue pairs across all NICs.
    pub fn num_qps(&self) -> usize {
        self.backend.num_qps()
    }

    /// Number of NICs this engine drives (= `rkeys.len()` of its descriptors).
    pub fn num_nics(&self) -> usize {
        self.backend.num_nics()
    }
}

#[cfg(test)]
mod tests {
    use super::{HandshakeMetadata, NicHandshake, RcEndpoint, RegionDescriptor};

    #[test]
    fn handshake_metadata_roundtrip() {
        let meta = HandshakeMetadata {
            nics: vec![NicHandshake {
                endpoints: vec![RcEndpoint {
                    gid: [7u8; 16],
                    lid: 0,
                    qp_num: 200,
                    psn: 0x1111,
                }],
            }],
        };
        let bytes = meta.to_bytes();
        let decoded = HandshakeMetadata::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.nics.len(), 1);
        assert_eq!(decoded.nics[0].endpoints, meta.nics[0].endpoints);
    }

    #[test]
    fn handshake_metadata_multi_nic_multi_qp_roundtrip() {
        let ep = |gid: u8, qp_num: u32| RcEndpoint {
            gid: [gid; 16],
            lid: 0,
            qp_num,
            psn: qp_num,
        };
        let meta = HandshakeMetadata {
            nics: vec![
                NicHandshake {
                    endpoints: vec![ep(1, 100), ep(1, 101)],
                },
                NicHandshake {
                    endpoints: vec![ep(2, 200), ep(2, 201)],
                },
            ],
        };
        let bytes = meta.to_bytes();
        let decoded = HandshakeMetadata::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.nics.len(), 2);
        assert_eq!(decoded.nics[0].endpoints.len(), 2);
        assert_eq!(decoded.nics[0].endpoints[1].qp_num, 101);
        assert_eq!(decoded.nics[1].endpoints[0].qp_num, 200);
    }

    #[test]
    fn handshake_metadata_rejects_garbage() {
        assert!(HandshakeMetadata::from_bytes(&[1, 2, 3]).is_err());
    }

    #[test]
    fn region_descriptor_roundtrip_and_contains() {
        let desc = RegionDescriptor {
            addr: 0x1000,
            len: 0x2000,
            rkeys: vec![10, 20],
        };
        let decoded = RegionDescriptor::from_bytes(&desc.to_bytes()).expect("decode");
        assert_eq!(decoded, desc);
        assert!(desc.contains(0x1000, 0x2000));
        assert!(desc.contains(0x2fff, 1));
        assert!(!desc.contains(0x0fff, 1));
        assert!(!desc.contains(0x2fff, 2));
        assert!(!desc.contains(u64::MAX, 1));
        assert!(RegionDescriptor::from_bytes(&[9]).is_err());
    }
}
