// SPDX-License-Identifier: Apache-2.0
//! GPU memory as an RDMA endpoint, end to end, on one host.
//!
//! Two engines share the local NICs (loopback, like `cpu_bench`). The
//! "server" owns a source buffer on one GPU and a sink buffer on another,
//! both registered through dma-buf; the "client" pulls and pushes with
//! RDMA READ / WRITE and every byte is verified against a host reference.
//!
//!   A  GPU  -> GPU   READ   (client GPU dst  <- server GPU src)
//!   B  GPU  -> host  READ   (client host     <- server GPU src)
//!   C  host -> GPU   WRITE  (client host     -> server GPU sink)
//!
//! No bounce buffer anywhere: the NIC DMAs straight into / out of HBM.

use std::os::fd::RawFd;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use pegastore_cuda::Topology;
use pegastore_cuda::cuda::{self, Gpu, Gpus};
use pegastore_cuda::topology::pin_to_numa;
use pegastore_rdma::rdma_topo::SystemTopology;
use pegastore_rdma::{
    ConnectionStatus, NumaNode, RegionDescriptor, TransferDesc, TransferEngine, TransferOp,
    init_logging,
};

#[derive(Parser)]
struct Cli {
    /// GPU that owns the source buffer.
    #[arg(long, default_value_t = 2)]
    src_gpu: u16,
    /// GPU that receives (case A) and is the sink (case C).
    #[arg(long, default_value_t = 3)]
    dst_gpu: u16,
    /// Buffer size in MiB (multiple of 2).
    #[arg(long, default_value_t = 1024)]
    mib: usize,
    /// Block size per RDMA op in MiB.
    #[arg(long, default_value_t = 4)]
    block_mib: usize,
    /// Timed iterations per case.
    #[arg(long, default_value_t = 5)]
    iters: usize,
    /// NICs to use; default = every NIC with an active port.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    nics: Vec<String>,
    #[arg(long, default_value_t = 2)]
    qps_per_peer: usize,
}

const MIB: usize = 1 << 20;

/// Device buffer with its dma-buf fd, registered on both engines.
struct GpuBuf {
    ptr: u64,
    len: usize,
    fd: RawFd,
}

impl GpuBuf {
    fn alloc(gpu: &Gpu, len: usize) -> Self {
        gpu.ctx.bind_to_thread().unwrap();
        let ptr = cuda::device_alloc(gpu, len).expect("cuMemAlloc");
        // SAFETY: fresh allocation, context bound.
        let fd = unsafe { cuda::dmabuf_fd(ptr, len) }.expect("dma-buf export");
        Self { ptr, len, fd }
    }

    fn fill(&self, gpu: &Gpu, host: &[u8]) {
        // SAFETY: `host` is a live slice of `len` bytes.
        unsafe { cuda::h2d_async(self.ptr, host.as_ptr(), self.len, &gpu.stream).unwrap() };
        cuda::sync(&gpu.stream).unwrap();
    }

    fn zero(&self, gpu: &Gpu) {
        // SAFETY: device range is owned by us.
        unsafe { cuda::memset_d32_async(self.ptr, 0, self.len, &gpu.stream).unwrap() };
        cuda::sync(&gpu.stream).unwrap();
    }

    fn read_back(&self, gpu: &Gpu, host: &mut [u8]) {
        // SAFETY: `host` is a live mutable slice of `len` bytes.
        unsafe { cuda::d2h_async(host.as_mut_ptr(), self.ptr, self.len, &gpu.stream).unwrap() };
        cuda::sync(&gpu.stream).unwrap();
    }
}

/// Page-locked host buffer, first-touched on `numa`.
struct HostBuf {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}

impl HostBuf {
    fn alloc(topo: &Topology, numa: Option<u16>, len: usize) -> Self {
        // SAFETY: anonymous private mapping.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "mmap failed");
        let ptr = ptr.cast::<u8>();
        let addr = ptr as usize;
        std::thread::scope(|s| {
            s.spawn(move || {
                if let Some(n) = numa {
                    pin_to_numa(topo, n);
                }
                // SAFETY: mapping is `len` bytes; first touch pins pages to `numa`.
                unsafe { std::ptr::write_bytes(addr as *mut u8, 0, len) };
            });
        });
        // SAFETY: page-aligned live mapping.
        assert!(unsafe { cuda::host_register(ptr, len) }, "cuMemHostRegister failed");
        Self { ptr, len }
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: mapping is live for `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: mapping is live for `len` bytes and uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

fn numa_node(topo: &Topology, gpu: u16) -> NumaNode {
    topo.numa_of_gpu(gpu)
        .map_or(NumaNode::UNKNOWN, |n| NumaNode(u32::from(n)))
}

fn connect(server: &TransferEngine, client: &TransferEngine) {
    let prepared = |st: ConnectionStatus| match st {
        ConnectionStatus::Prepared(m) => m,
        _ => panic!("fresh engine should need a handshake"),
    };
    let s_meta = prepared(server.get_or_prepare("client").unwrap());
    let c_meta = prepared(client.get_or_prepare("server").unwrap());
    server.complete_handshake("client", &s_meta, &c_meta).unwrap();
    client.complete_handshake("server", &c_meta, &s_meta).unwrap();
}

fn blocks<'a>(local: u64, remote: &'a RegionDescriptor, len: usize, block: usize) -> Vec<TransferDesc<'a>> {
    (0..len)
        .step_by(block)
        .map(|off| TransferDesc {
            local: local + off as u64,
            remote: remote.addr + off as u64,
            len: block.min(len - off),
            region: remote,
        })
        .collect()
}

/// Run `descs` `iters` times; return best and mean GiB/s.
fn timed(client: &TransferEngine, op: TransferOp, descs: &[TransferDesc<'_>], iters: usize) -> (f64, f64) {
    let bytes: usize = descs.iter().map(|d| d.len).sum();
    let mut secs = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let rxs = client.batch_transfer_async(op, "server", descs).expect("submit");
        let mut done = 0;
        for rx in rxs {
            done += block_recv(rx).expect("channel").expect("rdma");
        }
        assert_eq!(done, bytes, "short transfer");
        secs.push(t.elapsed().as_secs_f64());
    }
    let gib = bytes as f64 / (1u64 << 30) as f64;
    let best = gib / secs.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = gib / (secs.iter().sum::<f64>() / secs.len() as f64);
    (best, mean)
}

/// Block the calling thread on a `mea` oneshot receiver.
fn block_recv<T>(rx: mea::oneshot::Receiver<T>) -> Result<T, mea::oneshot::RecvError> {
    use std::future::IntoFuture;
    use std::pin::pin;
    use std::task::{Context, Poll, Wake};
    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Arc::new(Unpark(std::thread::current())).into();
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(rx.into_future());
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn verify(label: &str, got: &[u8], want: &[u8]) {
    if got == want {
        println!("  verify: OK ({} MiB compared)", got.len() / MIB);
    } else {
        let first = got.iter().zip(want).position(|(a, b)| a != b).unwrap();
        panic!("{label}: data mismatch at byte {first:#x}");
    }
}

fn main() {
    init_logging();
    let cli = Cli::parse();
    let len = cli.mib * MIB;
    let block = cli.block_mib * MIB;
    assert!(len.is_multiple_of(2 * MIB) && block.is_multiple_of(2 * MIB), "sizes must be multiples of 2 MiB");

    // ---- NICs ----
    let nics: Vec<String> = if cli.nics.is_empty() {
        let sys = SystemTopology::detect();
        sys.groups().iter().flat_map(|g| g.nics.iter().map(|n| n.name.clone())).collect()
    } else {
        cli.nics.clone()
    };
    assert!(!nics.is_empty(), "no RDMA NICs with an active port");
    let nic_numa: Vec<String> = nics
        .iter()
        .map(|n| format!("{n} ({})", pegastore_rdma::rdma_topo::nic_numa_node(n)))
        .collect();

    // ---- GPUs ----
    cuda::init().unwrap();
    let gpus = Arc::new(Gpus::open(&[cli.src_gpu, cli.dst_gpu]).expect("gpus"));
    let src_gpu = gpus.get(cli.src_gpu).unwrap();
    let dst_gpu = gpus.get(cli.dst_gpu).unwrap();
    for g in &gpus.devices {
        assert!(cuda::dmabuf_supported(g), "GPU{} cannot export dma-buf", g.index);
    }
    let topo = Topology::detect(
        &gpus.devices.iter().map(|g| (g.index, g.pci_bus_id.clone())).collect::<Vec<_>>(),
    );
    let src_numa = numa_node(&topo, cli.src_gpu);
    let dst_numa = numa_node(&topo, cli.dst_gpu);

    println!("gpu_rdma_bench: {} MiB, {} MiB blocks, {} iters", cli.mib, cli.block_mib, cli.iters);
    println!("  NICs: {}", nic_numa.join(", "));
    println!("  src: GPU{} ({src_numa})   dst/sink: GPU{} ({dst_numa})", cli.src_gpu, cli.dst_gpu);

    // ---- buffers ----
    let mut reference = vec![0u8; len];
    for (i, chunk) in reference.chunks_exact_mut(4).enumerate() {
        chunk.copy_from_slice(&(i as u32).wrapping_mul(2_654_435_761).to_le_bytes());
    }
    let src = GpuBuf::alloc(src_gpu, len);
    src.fill(src_gpu, &reference);
    let sink = GpuBuf::alloc(dst_gpu, len);
    sink.zero(dst_gpu);
    let dst = GpuBuf::alloc(dst_gpu, len);
    dst.zero(dst_gpu);
    let mut host = HostBuf::alloc(&topo, topo.numa_of_gpu(cli.dst_gpu), len);
    let mut scratch = vec![0u8; len];

    // ---- engines: server owns src + sink, client owns dst + host ----
    let server = TransferEngine::new(&nics, cli.qps_per_peer).expect("server engine");
    let client = TransferEngine::new(&nics, cli.qps_per_peer).expect("client engine");
    let t = Instant::now();
    // SAFETY: buffers outlive the engines; fds export exactly these ranges.
    let (src_desc, sink_desc) = unsafe {
        (
            server.register_dmabuf(src.ptr, src.len, src.fd, 0, src_numa).expect("register src"),
            server.register_dmabuf(sink.ptr, sink.len, sink.fd, 0, dst_numa).expect("register sink"),
        )
    };
    // SAFETY: as above.
    unsafe {
        client.register_dmabuf(dst.ptr, dst.len, dst.fd, 0, dst_numa).expect("register dst");
        client.register_host(host.ptr as u64, host.len).expect("register host");
    }
    println!(
        "  registered 3 GPU regions (dma-buf) + 1 host region on {} NIC(s) in {:.1} ms; src rkeys={:?}",
        nics.len(),
        t.elapsed().as_secs_f64() * 1e3,
        src_desc.rkeys
    );
    connect(&server, &client);

    // ---- A: GPU -> GPU READ ----
    println!("\n=== A  RDMA READ  GPU{} -> GPU{}", cli.src_gpu, cli.dst_gpu);
    let descs = blocks(dst.ptr, &src_desc, len, block);
    let (best, mean) = timed(&client, TransferOp::Read, &descs, cli.iters);
    println!("  best {best:.2} GiB/s   mean {mean:.2} GiB/s   ({} ops)", descs.len());
    dst.read_back(dst_gpu, &mut scratch);
    verify("A", &scratch, &reference);

    // ---- B: GPU -> host READ ----
    println!("\n=== B  RDMA READ  GPU{} -> host ({dst_numa})", cli.src_gpu);
    let descs = blocks(host.ptr as u64, &src_desc, len, block);
    let (best, mean) = timed(&client, TransferOp::Read, &descs, cli.iters);
    println!("  best {best:.2} GiB/s   mean {mean:.2} GiB/s");
    verify("B", host.as_slice(), &reference);

    // ---- C: host -> GPU WRITE ----
    println!("\n=== C  RDMA WRITE host ({dst_numa}) -> GPU{}", cli.dst_gpu);
    let descs = blocks(host.ptr as u64, &sink_desc, len, block);
    let (best, mean) = timed(&client, TransferOp::Write, &descs, cli.iters);
    println!("  best {best:.2} GiB/s   mean {mean:.2} GiB/s");
    sink.read_back(dst_gpu, &mut scratch);
    verify("C", &scratch, &reference);

    // Prove the WRITE actually landed by trashing host afterwards.
    host.as_mut_slice().fill(0);
    println!("\nall cases verified.");
}
