//! One put. Every GPU on the node. The store picks the wire.
//!
//! A weight shard (N slots × S GiB) is produced on GPU0 and `put` once with
//! `Placement::Each([cpu0, cpu1])`. Then:
//!
//!   A. every GPU `get`s it concurrently — each is served from the DRAM
//!      replica on its own socket;
//!   B. the same shard is `put` again pinned to one socket only, and every
//!      GPU `get`s it — the far socket's GPUs pay the cross-socket path;
//!   C. GPU0 `publish`es the copy it already holds, and the other GPUs `get`
//!      again — now served from GPU0 over NVLink, DRAM untouched.
//!
//! Usage: weights_fanout [--gib <per slot, default 1>] [--slots <n, default 4>]
//!                       [--pool-gib <per NUMA, default 3*slots*gib>]

use std::future::IntoFuture;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::executor::block_on;
use pegastore::{Device, Key, MemoryRegion, OpGet, Placement, PutSlot, Retention, SlotIdx, SlotSpec, Store, Tier};
use pegastore_cuda::cuda::{self, Gpus};
use pegastore_cuda::topology::pin_to_numa;
use pegastore_cuda::{Local, Topology};

const GIB: usize = 1 << 30;

struct Args {
    gib: usize,
    slots: usize,
    pool_gib: Option<usize>,
}

fn parse_args() -> Args {
    let mut a = Args {
        gib: 1,
        slots: 4,
        pool_gib: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_default();
        match k.as_str() {
            "--gib" => a.gib = v.parse().expect("--gib"),
            "--slots" => a.slots = v.parse().expect("--slots"),
            "--pool-gib" => a.pool_gib = Some(v.parse().expect("--pool-gib")),
            _ => panic!("unknown arg {k}"),
        }
    }
    a
}

/// A device allocation registered with the store.
struct GpuBuf {
    gpu: u16,
    ptr: u64,
    len: usize,
    region: Option<MemoryRegion>,
}

impl GpuBuf {
    fn new(store: &Store, gpus: &Gpus, gpu: u16, len: usize) -> Self {
        let g = gpus.get(gpu).unwrap();
        let ptr = cuda::device_alloc(g, len).expect("cuMemAlloc");
        // SAFETY: the allocation lives as long as this GpuBuf (leaked at exit).
        let region = unsafe { store.register(ptr as *mut u8, len as u64, Device::gpu(gpu)) }.expect("register");
        Self {
            gpu,
            ptr,
            len,
            region: Some(region),
        }
    }

    fn iov(&self) -> [pegastore::Iov<'_>; 1] {
        [self.region.as_ref().unwrap().iov_all()]
    }
}

fn fill(gpus: &Gpus, b: &GpuBuf, pattern: u32) {
    let g = gpus.get(b.gpu).unwrap();
    // SAFETY: b.ptr/len is our own allocation.
    unsafe { cuda::memset_d32_async(b.ptr, pattern, b.len, &g.stream).unwrap() };
    cuda::sync(&g.stream).unwrap();
}

/// Check a 4 MiB sample at the start, middle and end of the buffer.
fn verify(gpus: &Gpus, b: &GpuBuf, pattern: u32) -> bool {
    let g = gpus.get(b.gpu).unwrap();
    let sample = (4 << 20).min(b.len);
    let mut host = vec![0u8; sample];
    for off in [0usize, b.len / 2 / 4 * 4, b.len - sample] {
        // SAFETY: within our allocation; host buffer sized to `sample`.
        unsafe { cuda::d2h_async(host.as_mut_ptr(), b.ptr + off as u64, sample, &g.stream).unwrap() };
        cuda::sync(&g.stream).unwrap();
        if host.chunks_exact(4).any(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]) != pattern) {
            return false;
        }
    }
    true
}

fn gbs(bytes: usize, d: Duration) -> f64 {
    bytes as f64 / d.as_secs_f64() / 1e9
}

struct GetResult {
    gpu: u16,
    elapsed: Duration,
    from: Vec<(Device, Tier)>,
    ok: bool,
}

/// Every listed GPU fetches all slots of `key` concurrently (one thread per
/// GPU, pinned to its socket), into freshly allocated device buffers.
#[allow(clippy::too_many_arguments)]
fn fan_out(
    store: &Store,
    gpus: &Gpus,
    topo: &Topology,
    gpu_ids: &[u16],
    key: &Key,
    slots: usize,
    slot_bytes: usize,
    patterns: &[u32],
    keep: bool,
) -> (Vec<GetResult>, Vec<Vec<GpuBuf>>) {
    let mut results = Vec::new();
    let mut kept = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = gpu_ids
            .iter()
            .map(|&gpu| {
                s.spawn(move || {
                    if let Some(n) = topo.numa_of_gpu(gpu) {
                        pin_to_numa(topo, n);
                    }
                    let bufs: Vec<GpuBuf> = (0..slots).map(|_| GpuBuf::new(store, gpus, gpu, slot_bytes)).collect();
                    let iovs: Vec<_> = bufs.iter().map(|b| b.iov()).collect();
                    let reqs: Vec<(Key, OpGet<'_>)> = iovs
                        .iter()
                        .enumerate()
                        .map(|(i, iov)| (key.clone(), OpGet::new(SlotIdx(i as u32), iov)))
                        .collect();
                    let t0 = Instant::now();
                    let mut from = vec![None; slots];
                    block_on(async {
                        let mut st = store.get_many(reqs);
                        while let Some((i, r)) = st.next().await {
                            let rp = r.expect("get");
                            from[i] = Some((rp.from.device, rp.from.tier));
                        }
                    });
                    let elapsed = t0.elapsed();
                    let ok = bufs.iter().zip(patterns).all(|(b, p)| verify(gpus, b, *p));
                    (
                        GetResult {
                            gpu,
                            elapsed,
                            from: from.into_iter().map(Option::unwrap).collect(),
                            ok,
                        },
                        bufs,
                    )
                })
            })
            .collect();
        for h in handles {
            let (r, bufs) = h.join().unwrap();
            results.push(r);
            if keep {
                kept.push(bufs);
            }
        }
    });
    results.sort_by_key(|r| r.gpu);
    (results, kept)
}

fn report(title: &str, rs: &[GetResult], total_bytes: usize, topo: &Topology) {
    println!("\n{title}");
    println!("  {:<6} {:<6} {:>10} {:>9}  served from", "gpu", "numa", "GB/s", "ms");
    for r in rs {
        let numa = topo.numa_of_gpu(r.gpu).map_or("?".into(), |n| n.to_string());
        let mut from: Vec<String> = r.from.iter().map(|(d, t)| format!("{d}/{t:?}")).collect();
        from.dedup();
        println!(
            "  {:<6} {:<6} {:>10.1} {:>9.1}  {}{}",
            r.gpu,
            numa,
            gbs(total_bytes, r.elapsed),
            r.elapsed.as_secs_f64() * 1e3,
            from.join(","),
            if r.ok { "" } else { "  ✗ DATA MISMATCH" }
        );
    }
    let slowest = rs.iter().map(|r| r.elapsed).max().unwrap();
    println!(
        "  aggregate: {:.1} GB/s ({} GPUs × {:.1} GiB in {:.1} ms)",
        gbs(total_bytes * rs.len(), slowest),
        rs.len(),
        total_bytes as f64 / GIB as f64,
        slowest.as_secs_f64() * 1e3
    );
}

fn main() {
    let args = parse_args();
    let slot_bytes = args.gib * GIB;
    let total = slot_bytes * args.slots;
    let pool = args.pool_gib.map_or(3 * total, |g| g * GIB);

    println!("pegastore · weights fan-out");
    println!("shard: {} slots × {} GiB = {} GiB", args.slots, args.gib, total / GIB);

    let t0 = Instant::now();
    let local = Local::builder().dram(0, pool).dram(1, pool).build().expect("local backend");
    let topo = local.topology().clone();
    println!("topology: {}", topo.describe());
    println!(
        "pinned pools: {} × {} GiB, ready in {:.2}s",
        local.pool_stats().len(),
        pool / GIB,
        t0.elapsed().as_secs_f64()
    );
    let store = Store::new(local);

    // Open our own handles to the same GPUs for producing/verifying data.
    let gpu_ids: Vec<u16> = topo.gpu_numa.keys().copied().collect::<Vec<_>>();
    let mut gpu_ids = gpu_ids;
    gpu_ids.sort_unstable();
    let gpus = Arc::new(Gpus::open(&gpu_ids).expect("gpus"));
    assert!(gpu_ids.len() >= 2, "need at least two GPUs");
    let (cpu0, cpu1) = (Device::cpu(topo.numa_nodes[0]), Device::cpu(*topo.numa_nodes.last().unwrap()));

    // Produce the shard on GPU0.
    let patterns: Vec<u32> = (0..args.slots).map(|i| 0xA5A5_0000 | i as u32).collect();
    let src: Vec<GpuBuf> = (0..args.slots).map(|_| GpuBuf::new(&store, &gpus, gpu_ids[0], slot_bytes)).collect();
    for (b, p) in src.iter().zip(&patterns) {
        fill(&gpus, b, *p);
    }
    let src_iovs: Vec<_> = src.iter().map(|b| b.iov()).collect();

    // ---- put once, one replica per socket ----
    let key_each = Key::from("weights/v1/shard0");
    let slots_each: Vec<PutSlot<'_>> = src_iovs
        .iter()
        .map(|iov| PutSlot::new(SlotSpec::new(slot_bytes as u64, Placement::Each(vec![cpu0, cpu1])), iov))
        .collect();
    let t = Instant::now();
    block_on(store.put_with(key_each.clone(), slots_each).retention(Retention::Explicit).into_future())
        .expect("put")
        .into_result()
        .expect("put slots");
    let d = t.elapsed();
    println!(
        "\nput  {:?}  Each([{cpu0}, {cpu1}])  from gpu{}: {:.1} ms, {:.1} GB/s written ({} GiB × 2 replicas)",
        key_each,
        gpu_ids[0],
        d.as_secs_f64() * 1e3,
        gbs(total * 2, d),
        total / GIB
    );
    let info = block_on(store.stat(std::slice::from_ref(&key_each))).unwrap().remove(0).unwrap();
    for (i, s) in info.slots.iter().enumerate() {
        let locs: Vec<String> = s.replicas.iter().map(|r| r.location.device.to_string()).collect();
        println!("  slot {i}: {} GiB @ [{}]", s.len as usize / GIB, locs.join(", "));
    }

    // ---- A: every GPU reads from its own socket ----
    let (ra, kept) = fan_out(&store, &gpus, &topo, &gpu_ids, &key_each, args.slots, slot_bytes, &patterns, true);
    report("A. get on every GPU — placement Each: each socket serves its own GPUs", &ra, total, &topo);

    // ---- B: same bytes pinned to one socket ----
    let key_one = Key::from("weights/v1/shard0.one-socket");
    let slots_one: Vec<PutSlot<'_>> = src_iovs
        .iter()
        .map(|iov| PutSlot::new(SlotSpec::new(slot_bytes as u64, Placement::Strict(cpu1)), iov))
        .collect();
    block_on(store.put_with(key_one.clone(), slots_one).retention(Retention::Explicit).into_future())
        .expect("put")
        .into_result()
        .expect("put slots");
    let (rb, _) = fan_out(&store, &gpus, &topo, &gpu_ids, &key_one, args.slots, slot_bytes, &patterns, false);
    report(
        &format!("B. get on every GPU — placement Strict({cpu1}): far-socket GPUs cross the interconnect"),
        &rb,
        total,
        &topo,
    );
    block_on(store.remove(std::slice::from_ref(&key_one))).unwrap();

    // ---- C: GPU0 publishes what it already holds; peers read over NVLink ----
    let gpu0_copy = &kept[0];
    for (i, b) in gpu0_copy.iter().enumerate() {
        block_on(store.publish(key_each.clone(), SlotIdx(i as u32), &b.iov())).expect("publish");
    }
    let peers: Vec<u16> = gpu_ids[1..].to_vec();
    let (rc, _) = fan_out(&store, &gpus, &topo, &peers, &key_each, args.slots, slot_bytes, &patterns, false);
    report(
        &format!("C. gpu{} published its copy — peers now read over NVLink, DRAM idle", gpu_ids[0]),
        &rc,
        total,
        &topo,
    );

    let stats = store
        .stat(std::slice::from_ref(&key_each));
    let info = block_on(stats).unwrap().remove(0).unwrap();
    let mut kinds: Vec<String> = info.slots[0]
        .replicas
        .iter()
        .map(|r| format!("{}/{:?}", r.location.device, r.location.tier))
        .collect();
    kinds.sort();
    println!("\nslot 0 replicas now: [{}]", kinds.join(", "));
    println!("done.");
}
