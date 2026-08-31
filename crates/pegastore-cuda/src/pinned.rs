//! NUMA-placed, DMA-pinned host pools.
//!
//! `mmap` + parallel first-touch from threads pinned to the target NUMA node
//! + `cuMemHostRegister`. Allocation is first-fit over a free list; frees
//! coalesce. Good enough for large slots; a log-structured extent allocator
//! replaces this once eviction policy matters.
#![allow(clippy::doc_lazy_continuation)]
//!
//! Derived from pegaflow's `pinned_mem.rs` (Apache-2.0).

use std::ptr;
use std::sync::{Arc, Mutex};

use pegastore::{Error, ErrorKind, Result};

use crate::cuda;
use crate::topology::{Topology, pin_to_numa};

const ALIGN: usize = 2 << 20; // 2 MiB: friendly to both DMA and THP

pub struct PinnedPool {
    numa: u16,
    base: *mut u8,
    len: usize,
    registered: bool,
    free: Mutex<Vec<(usize, usize)>>, // (offset, len), sorted by offset
}

// SAFETY: the mapping is process-wide; the free list is behind a mutex.
unsafe impl Send for PinnedPool {}
unsafe impl Sync for PinnedPool {}

impl PinnedPool {
    pub fn new(topo: &Topology, numa: u16, bytes: usize) -> Result<Arc<Self>> {
        let len = bytes.div_ceil(ALIGN) * ALIGN;
        // SAFETY: anonymous private mapping; result checked.
        let base = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(Error::new(
                ErrorKind::NoSpace,
                format!("mmap {len} bytes: {}", std::io::Error::last_os_error()),
            ));
        }
        let base = base as *mut u8;
        first_touch(topo, numa, base, len);
        // SAFETY: base/len is a live mapping.
        let registered = unsafe { cuda::host_register(base, len) };
        if !registered {
            tracing::warn!(numa, len, "cuMemHostRegister failed; pool is pageable");
            eprintln!("warning: cuMemHostRegister({len} bytes, numa{numa}) failed; pool is pageable (check ulimit -l)");
        }
        Ok(Arc::new(Self {
            numa,
            base,
            len,
            registered,
            free: Mutex::new(vec![(0, len)]),
        }))
    }

    pub fn numa(&self) -> u16 {
        self.numa
    }

    pub fn capacity(&self) -> usize {
        self.len
    }

    pub fn free_bytes(&self) -> usize {
        self.free.lock().unwrap().iter().map(|(_, l)| l).sum()
    }

    /// Largest single allocation currently possible.
    pub fn largest_free(&self) -> usize {
        self.free.lock().unwrap().iter().map(|(_, l)| *l).max().unwrap_or(0)
    }

    pub fn alloc(self: &Arc<Self>, bytes: usize) -> Option<PinnedBuf> {
        let need = bytes.div_ceil(ALIGN) * ALIGN;
        let mut free = self.free.lock().unwrap();
        let i = free.iter().position(|(_, l)| *l >= need)?;
        let (off, l) = free[i];
        if l == need {
            free.remove(i);
        } else {
            free[i] = (off + need, l - need);
        }
        Some(PinnedBuf {
            pool: self.clone(),
            off,
            reserved: need,
            len: bytes,
        })
    }

    fn release(&self, off: usize, len: usize) {
        let mut free = self.free.lock().unwrap();
        let i = free.partition_point(|(o, _)| *o < off);
        free.insert(i, (off, len));
        // Coalesce with neighbours.
        if i + 1 < free.len() && free[i].0 + free[i].1 == free[i + 1].0 {
            free[i].1 += free[i + 1].1;
            free.remove(i + 1);
        }
        if i > 0 && free[i - 1].0 + free[i - 1].1 == free[i].0 {
            free[i - 1].1 += free[i].1;
            free.remove(i);
        }
    }
}

impl Drop for PinnedPool {
    fn drop(&mut self) {
        // SAFETY: registered/mapped by us with these exact parameters.
        unsafe {
            if self.registered {
                cuda::host_unregister(self.base);
            }
            libc::munmap(self.base as *mut libc::c_void, self.len);
        }
    }
}

/// An allocation inside a pool; returns its range on drop. Holding one keeps
/// the bytes alive through in-flight transfers even after eviction.
pub struct PinnedBuf {
    pool: Arc<PinnedPool>,
    off: usize,
    /// Reserved (aligned) length returned to the pool on drop.
    reserved: usize,
    /// Requested length; what `len()` reports.
    len: usize,
}

impl PinnedBuf {
    pub fn as_ptr(&self) -> *mut u8 {
        self.pool.base.wrapping_add(self.off)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn numa(&self) -> u16 {
        self.pool.numa
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        self.pool.release(self.off, self.reserved);
    }
}

/// Touch one byte per page from threads pinned to `numa` so first-touch
/// policy places every page on that node.
fn first_touch(topo: &Topology, numa: u16, base: *mut u8, len: usize) {
    let page = 4096usize;
    let threads = topo
        .cpus_by_numa
        .get(&numa)
        .map_or(8, |c| c.len().clamp(1, 32));
    let chunk = len.div_ceil(threads).div_ceil(page) * page;
    let base_addr = base as usize;
    std::thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                pin_to_numa(topo, numa);
                let start = t * chunk;
                if start >= len {
                    return;
                }
                let end = (start + chunk).min(len);
                let mut off = start;
                while off < end {
                    // SAFETY: within the mapping.
                    unsafe { ptr::write_volatile((base_addr + off) as *mut u8, 0) };
                    off += page;
                }
            });
        }
    });
}
