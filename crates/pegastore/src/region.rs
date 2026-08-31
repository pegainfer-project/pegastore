//! Registered memory and I/O vectors. All bytes enter and leave the store
//! through an `Iov` into a `MemoryRegion`; there is no unregistered slow path.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::Device;

static NEXT_REGION_ID: AtomicU64 = AtomicU64::new(1);

/// A user buffer registered with a backend. Dropping it unregisters (and
/// retires any `External` replicas published from it).
pub struct MemoryRegion {
    id: u64,
    addr: usize,
    len: u64,
    device: Device,
    /// Backend-private state (RDMA MR, CUDA IPC handle, shm mapping, ...).
    /// Its `Drop` performs the unregistration.
    _handle: Box<dyn Any + Send + Sync>,
}

// SAFETY: the region only carries an address; the backend that created it is
// responsible for making the memory usable from any thread.
unsafe impl Send for MemoryRegion {}
unsafe impl Sync for MemoryRegion {}

impl MemoryRegion {
    /// Allocate a fresh region id. Backends that need the id before building
    /// the handle (to tie the handle's `Drop` to it) call this first.
    pub fn next_id() -> u64 {
        NEXT_REGION_ID.fetch_add(1, Ordering::Relaxed)
    }

    /// # Safety
    /// `[addr, addr + len)` must stay valid and accessible according to
    /// `device` for the lifetime of the returned region.
    pub unsafe fn with_id(
        id: u64,
        addr: *mut u8,
        len: u64,
        device: Device,
        handle: Box<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            id,
            addr: addr as usize,
            len,
            device,
            _handle: handle,
        }
    }

    /// # Safety
    /// See [`MemoryRegion::with_id`].
    pub unsafe fn new(
        addr: *mut u8,
        len: u64,
        device: Device,
        handle: Box<dyn Any + Send + Sync>,
    ) -> Self {
        // SAFETY: forwarded to the caller's contract.
        unsafe { Self::with_id(Self::next_id(), addr, len, device, handle) }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.addr as *mut u8
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// A sub-range of this region. Bounds are checked by `validate_iovs`,
    /// not here.
    pub fn iov(&self, offset: u64, len: u64) -> Iov<'_> {
        Iov {
            region: self,
            offset,
            len,
        }
    }

    pub fn iov_all(&self) -> Iov<'_> {
        self.iov(0, self.len)
    }
}

impl fmt::Debug for MemoryRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryRegion")
            .field("id", &self.id)
            .field("addr", &format_args!("{:#x}", self.addr))
            .field("len", &self.len)
            .field("device", &self.device)
            .finish()
    }
}

/// One contiguous segment of a registered region.
#[derive(Clone, Copy)]
pub struct Iov<'a> {
    pub region: &'a MemoryRegion,
    pub offset: u64,
    pub len: u64,
}

impl Iov<'_> {
    pub fn as_ptr(&self) -> *mut u8 {
        self.region.as_ptr().wrapping_add(self.offset as usize)
    }

    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.len)
    }

    pub fn in_bounds(&self) -> bool {
        self.offset.checked_add(self.len).is_some_and(|e| e <= self.region.len())
    }

    pub fn device(&self) -> Device {
        self.region.device()
    }
}

impl fmt::Debug for Iov<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iov")
            .field("region", &self.region.id())
            .field("device", &self.region.device())
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}
