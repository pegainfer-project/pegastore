//! Thin CUDA driver helpers: contexts + streams per GPU, directional copies,
//! host registration. Everything here is synchronous; the backend decides
//! what to overlap.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaStream};
use pegastore::{Error, ErrorKind, Result};

pub fn check(res: sys::CUresult, what: &'static str) -> Result<()> {
    if res == sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::new(ErrorKind::Unexpected, format!("{what} failed: {res:?}")))
    }
}

/// One context + one copy stream per visible GPU. Peer access enabled
/// between every pair that supports it.
pub struct Gpus {
    pub devices: Vec<Gpu>,
}

pub struct Gpu {
    pub index: u16,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub pci_bus_id: String,
}

impl Gpus {
    pub fn open(indices: &[u16]) -> Result<Self> {
        let mut devices = Vec::with_capacity(indices.len());
        for &i in indices {
            let ctx = CudaContext::new(i as usize)
                .map_err(|e| Error::new(ErrorKind::Unavailable, format!("cuda device {i}: {e}")))?;
            let stream = ctx
                .new_stream()
                .map_err(|e| Error::new(ErrorKind::Unexpected, format!("new_stream gpu {i}: {e}")))?;
            let pci_bus_id = pci_bus_id(i)?;
            devices.push(Gpu {
                index: i,
                ctx,
                stream,
                pci_bus_id,
            });
        }
        // Enable peer access both ways; failures are non-fatal (copies fall
        // back to staging through the host).
        for a in &devices {
            a.ctx.bind_to_thread().ok();
            for b in &devices {
                if a.index == b.index {
                    continue;
                }
                // SAFETY: both contexts are live primary contexts.
                let r = unsafe { sys::cuCtxEnablePeerAccess(b.ctx.cu_ctx(), 0) };
                if r != sys::cudaError_enum::CUDA_SUCCESS
                    && r != sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
                {
                    tracing::warn!(from = a.index, to = b.index, ?r, "peer access not enabled");
                }
            }
        }
        Ok(Self { devices })
    }

    pub fn get(&self, index: u16) -> Option<&Gpu> {
        self.devices.iter().find(|g| g.index == index)
    }

    pub fn by_index(&self) -> HashMap<u16, &Gpu> {
        self.devices.iter().map(|g| (g.index, g)).collect()
    }
}

fn pci_bus_id(index: u16) -> Result<String> {
    let mut dev: sys::CUdevice = 0;
    // SAFETY: plain driver queries with valid out-pointers.
    unsafe {
        check(sys::cuDeviceGet(&mut dev, index as i32), "cuDeviceGet")?;
        let mut buf = [0u8; 32];
        check(
            sys::cuDeviceGetPCIBusId(buf.as_mut_ptr().cast::<std::ffi::c_char>(), buf.len() as i32, dev),
            "cuDeviceGetPCIBusId",
        )?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).to_ascii_lowercase())
    }
}

/// Copy `len` bytes host → device on `stream` (asynchronous).
///
/// # Safety
/// `dst` is a device address in `stream`'s context, `src` is host memory
/// that stays valid until the stream is synchronized.
pub unsafe fn h2d_async(dst: u64, src: *const u8, len: usize, stream: &CudaStream) -> Result<()> {
    // SAFETY: forwarded.
    check(
        unsafe { sys::cuMemcpyHtoDAsync_v2(dst, src as *const c_void, len, stream.cu_stream()) },
        "cuMemcpyHtoDAsync",
    )
}

/// # Safety
/// Mirror of [`h2d_async`].
pub unsafe fn d2h_async(dst: *mut u8, src: u64, len: usize, stream: &CudaStream) -> Result<()> {
    // SAFETY: forwarded.
    check(
        unsafe { sys::cuMemcpyDtoHAsync_v2(dst as *mut c_void, src, len, stream.cu_stream()) },
        "cuMemcpyDtoHAsync",
    )
}

/// Device → device across contexts (NVLink when peer access is enabled).
///
/// # Safety
/// Both addresses are valid device allocations in the given contexts.
pub unsafe fn peer_async(
    dst: u64,
    dst_ctx: &CudaContext,
    src: u64,
    src_ctx: &CudaContext,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    // SAFETY: forwarded.
    check(
        unsafe {
            sys::cuMemcpyPeerAsync(dst, dst_ctx.cu_ctx(), src, src_ctx.cu_ctx(), len, stream.cu_stream())
        },
        "cuMemcpyPeerAsync",
    )
}

pub fn sync(stream: &CudaStream) -> Result<()> {
    stream
        .synchronize()
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("stream sync: {e}")))
}

/// Pin host memory for DMA. Returns false if the driver refused (already
/// registered, not page aligned, ...); the memory stays usable, just slower.
///
/// # Safety
/// `[ptr, ptr + len)` is a valid mapping that outlives the registration.
pub unsafe fn host_register(ptr: *mut u8, len: usize) -> bool {
    // SAFETY: forwarded.
    let r = unsafe {
        sys::cuMemHostRegister_v2(ptr as *mut c_void, len, sys::CU_MEMHOSTREGISTER_PORTABLE)
    };
    r == sys::cudaError_enum::CUDA_SUCCESS
}

/// # Safety
/// `ptr` was registered with [`host_register`] and returned true.
pub unsafe fn host_unregister(ptr: *mut u8) {
    // SAFETY: forwarded.
    let _ = unsafe { sys::cuMemHostUnregister(ptr as *mut c_void) };
}

/// Allocate device memory in `gpu`'s context.
pub fn device_alloc(gpu: &Gpu, len: usize) -> Result<u64> {
    gpu.ctx
        .bind_to_thread()
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("bind ctx: {e}")))?;
    let mut ptr: sys::CUdeviceptr = 0;
    // SAFETY: valid out-pointer; context bound above.
    check(unsafe { sys::cuMemAlloc_v2(&mut ptr, len) }, "cuMemAlloc")?;
    Ok(ptr)
}

/// # Safety
/// `ptr` came from [`device_alloc`] on `gpu`.
pub unsafe fn device_free(gpu: &Gpu, ptr: u64) {
    gpu.ctx.bind_to_thread().ok();
    // SAFETY: forwarded.
    let _ = unsafe { sys::cuMemFree_v2(ptr) };
}

/// Fill device memory with a 32-bit pattern (asynchronous).
///
/// # Safety
/// `ptr` is a valid device allocation of at least `len` bytes; `len % 4 == 0`.
pub unsafe fn memset_d32_async(ptr: u64, value: u32, len: usize, stream: &CudaStream) -> Result<()> {
    // SAFETY: forwarded.
    check(
        unsafe { sys::cuMemsetD32Async(ptr, value, len / 4, stream.cu_stream()) },
        "cuMemsetD32Async",
    )
}

pub fn device_count() -> Result<u16> {
    let mut n: i32 = 0;
    // SAFETY: valid out-pointer.
    check(unsafe { sys::cuDeviceGetCount(&mut n) }, "cuDeviceGetCount")?;
    Ok(n.max(0) as u16)
}

pub fn init() -> Result<()> {
    // SAFETY: cuInit(0) is always sound.
    check(unsafe { sys::cuInit(0) }, "cuInit")
}
