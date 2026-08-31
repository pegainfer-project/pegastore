# Lessons

Design notes distilled from building the first backends and from reading two
production-shaped consumers: `op-cache` (single-node warm restart through a
pinned-RAM daemon) and `wc-ps` (fleet-wide R=1 checkpoint cache with RDMA and
NVLink broadcast). Recorded so the next interface discussion starts here, not
from zero.

## 1. The store must not know what a tensor is

Both consumers agree by their behavior:

- `op-cache` carries `dtype`/`shape` per tensor but uses them for validation
  only; only the byte range is load-bearing. Strides are not carried at all;
  contiguity is required instead.
- `wc-ps` keeps 118k–497k tensor descriptors in a sidecar manifest and stores
  only opaque ~4 GiB buckets. The engine's own `load_weights` picks its shard
  out of the full stream.

So: no per-slot tag, no dtype, no names in pegastore. **The manifest is an
ordinary object** under the same key prefix, with the same placement and
retention. `Placement::Each` suits it — small, wanted on every node.

## 2. Slot granularity is transport granularity

GLM-5.2-FP8 has 118,629 tensors; Kimi-K3 has 497,220. If slot = tensor, a
restore is a few hundred thousand async operations and the per-op overhead
alone exceeds the 0.2–0.3 s that `op-cache` achieves per rank. Both consumers
cut at tens of MiB to GiB: allocator segments (`op-cache`) or buckets
(`wc-ps`). Names → (slot, offset, len) live in the manifest. Ranged `get`
plus scatter `Iov`s put tensor bytes at their final addresses without a
landing buffer.

## 3. Only add semantics the user cannot build outside

The filter for any proposed addition: can a consumer implement it in its own
code on top of the existing seven operations? Rejected by that filter:

| proposal | why the user can do it |
|---|---|
| per-slot tag, `list(prefix)` | manifest object |
| bucket packing, HRW placement | `Placement::Strict` expresses the result; the hash is theirs |
| windowed streaming `get` | `get` is per-slot and async; a semaphore or ring around it is trivial |
| same-tray fan-out layer | NCCL broadcast, or `publish` + distance |
| per-slot checksum | manifest, verify after `get` |
| `put` from a file descriptor | one extra memcpy through user memory; disk (15–20 GB/s) is the bottleneck, not the copy |
| "object complete = hit" | `stat`, all slots present; idempotent refill fixes the rest |

What survived the filter is in §4 and §6.

## 4. Addresses must cross process boundaries

The warm-restart case forces the pinned pool out of the engine process. Then
every `get` destination and every snapshot source is another process's
memory. A raw pointer cannot express that; only the store can turn a foreign
address into something its copy engines can reach.

Two directions were weighed:

- **push** (`op-cache`): the consumer exports its GPU allocation (cudaIpc on
  the allocation base, or a cuMem shareable fd for expandable segments), the
  daemon imports it and writes. The store imports foreign memory and must
  track its lifetime.
- **pull** (`wc-ps`, `publish`, RDMA READ): the store exports, the consumer
  copies into its own memory in its own stream. The store never touches
  foreign memory.

Pull wins on lifetime and on uniformity (same shape as NVLink peer read and
RDMA READ), but host memory cannot be exported with cudaIpc and consumer-side
`cudaHostRegister` walks pages at ~25 GB/s (96 GB in 3.9 s, 620 GB in 25 s —
unacceptable per restart). Resolution: **the daemon owns a small HBM ring per
GPU, exported once via cudaIpc.** It H2Ds a slot into the ring at the C2C
ceiling; the consumer D2Ds ring → final address at HBM speed. No bandwidth is
lost; the store still only exports.

Open item: `cuMemCreate(CU_MEM_LOCATION_TYPE_HOST_NUMA)` + POSIX-fd export
would let consumers map the pool itself with no page walk and gives NUMA
placement as an allocation parameter. Needs a 50-line test on a GB300 tray.

## 5. The ring needs no new interface

Once the consumer has imported the ring, it is an ordinary GPU address in the
consumer's process and can be `register`ed like any other region. Then:

```rust
let window = client.window(gpu)?;                 // import the daemon's ring (client helper)
store.get(key, slot, &[window.iov(k * SLOT, len)]).await?;  // the existing get
// consumer D2Ds window → final dst on its own stream, then issues the next get for slot k
```

The `get` completing is the "slot filled" signal; the next `get` for the same
slot is the "slot free" signal. No IPC events. The daemon recognizes the
region as its own export and skips import. A consumer that prefers its own
HBM as the ring registers that instead — same code path, push shape.

The write path stays push: a snapshot is one-time, reads are the hot path,
and a pull-shaped `put` (allocate, export, write, commit) is the `seal` we
removed under another name.

## 6. Session semantics, stated

- `Retention::Explicit` objects belong to no connection; they outlive the
  client. This is what makes warm restart possible.
- Registrations and `publish`ed External replicas belong to the connection
  that made them. When it ends — `unregister`, drop, or process death — the
  store retires them, **and retirement waits for in-flight operations**. A
  synchronous retire while a peer is mid-NVLink-read is a use-after-free.
- A Cache-retained replica that a consumer has mapped (cudaIpc/cuMem) cannot
  be evicted until the mapping closes. Explicit replicas never face this.

## 7. What a daemon deployment consists of

`Access` is unchanged. The work is a servicer pair plus one helper:

- **`DaemonClient: Access`** in the engine process. `register` exports the
  region (cudaIpc / cuMem fd / memfd) and sends the handle; every other
  operation is an RPC carrying `(key, slot, region_id, offset, len)`. Bytes
  never cross the RPC.
- **`pegastored`** holds a `Local`, imports client regions, owns per-GPU rings,
  and unregisters everything a connection owned when it drops.
- **`DaemonClient::window(gpu) -> MemoryRegion`**: import the ring. Not in
  `Access` — it is specific to this deployment.

Nodes that contribute only DRAM and a NIC must run without CUDA:
`pegastore-cuda` is optional to the DRAM tier and to the RDMA server side.

## 8. Loading a model into one tray of an NVL72

Per-GPU NVLink ingress (~900 GB/s) is the floor: a GPU that must receive M
bytes cannot finish before M / 900 GB/s. Everything else is arranged around
that.

- Store one copy, striped over the rack: slot i on tray (i mod 18), in the
  DRAM of the Grace attached to the GPU that will serve it. 755 GB costs
  42 GB per tray. SSD holds the second copy.
- Source side is never the limit past 4 trays: each tray's DRAM egress
  (~1 TB/s) times 4 already exceeds the consumers' 3.6 TB/s aggregate ingress.
  Eighteen trays buy footprint, not bandwidth.
- If each consumer needs the *full* stream, pull M/4 from remote and take
  3M/4 from the three neighbors over intra-tray NVLink (`publish` + distance,
  or a broadcast collective). Same per-GPU ingress, one quarter the rack
  traffic, one read per byte from production DRAM.
- Work-queue slot assignment, not static source binding: a slow or dead
  source only slows its own slots.
- Firing async reads at every peer simultaneously wedged the fabric in
  `wc-ps`; per-peer queue depth is a backend policy, not an interface.

At 200 Gbps the NIC hides every NUMA effect: GPU0/1 (cross-socket from
`mlx5_bond_0`) land within 1% of GPU2/3. The cross-socket penalty measured
elsewhere (H2D 211 vs 126 GB/s; restore 101 vs 72 GiB/s) reappears at
400/800G.

## 9. RDMA engine

- Memory registration must be independent of the handshake. v1 snapshotted
  MRs at connect time, so memory registered later was invisible to peers.
  `RegionDescriptor { addr, len, rkeys }` travels with the replica instead.
- NIC discovery by port state, not by name. `name.contains("bond")` hid the
  only live NIC on a RoCE host and listed four DOWN IB ports.
- dma-buf registration of HBM works end to end on GB300: GPU→GPU, GPU→DRAM,
  DRAM→GPU all at bond line rate (22.9–23.0 GiB/s), no bounce, memcmp clean.
  Registering 3 GiB of HBM took 2.7 ms.

## 10. Amdahl

`op-cache` took the weights phase from 212 s to 1.2 s. After that, engine
`init` (13 s) and Python `load_weights` (10–60 s at 100k+ tensors) dominate.
The store's share of a warm start is already a few seconds; the next order of
magnitude comes from the loader scattering straight from the manifest to
final addresses, not from the store.

## 11. Layering

- **core** (`pegastore`): slots, placement, addresses, seven operations.
  Stable; knows nothing above bytes.
- **consumer code** (`weights-cache`, KV connectors): packing, manifests,
  hashing, parallelism, engine snapshot points. Owned by the consumer; the
  store does not ship opinions here unless asked.
- **deployment** (`pegastored`, `DaemonClient`, `Remote`): how the same seven
  operations reach another process or node.

Every difference between the two consumers landed in the second and third
layers. The object model did not move.

## Reference numbers (tray03, 4× GB300, 2026-08)

| path | figure |
|---|---|
| `Each` placement fan-out, 4 GPUs | 693 GB/s aggregate |
| `Strict` one socket, 4 GPUs | 253 GB/s aggregate |
| `publish` GPU 0, NVLink peers | 794 GB/s aggregate |
| H2D NUMA-local / cross-socket (op-cache) | 211 / 126 GB/s |
| RDMA loopback, `mlx5_bond_0` (2×100GbE) | 196.6 Gbps READ and WRITE |
| dma-buf GPU↔GPU / GPU↔DRAM over RDMA | 22.9–23.0 GiB/s |
| `cudaHostRegister` page walk | ~25 GB/s |
| 755 GB into 4 GPUs, ideal | 0.21 s (189 GB/GPU ÷ 900 GB/s) |
