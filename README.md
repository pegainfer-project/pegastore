# pegastore

**Why does a store hand you bytes and keep the address to itself?**

On one GB300 tray, DRAM on the far socket is 3× slower than DRAM on the near one.
A neighbouring GPU over NVLink is 2× faster than either.
The engine knows this. The store pretends not to.

So every AI system rebuilds placement by hand: a KV connector here, a weight
loader there, a checkpoint shipper, each with its own copy pipeline, its own
NUMA folklore, its own RDMA plumbing. Thousands of lines, per framework, to
move the same bytes to the same place.

---

## The world after

One `put`.
Every GPU on the node reads it at wire speed, from the copy nearest to it.
A reader becomes a source with one more call.
Weights, KV cache, hidden states — one API, no connectors.

Placement is a parameter, not a subsystem.

---

## Location is the API

A value is a set of **slots**.
Each slot is placed on a **device** — `cpu:0`, `cpu:1`, `gpu:3`.
A read is served from the **nearest** replica.

Objects are immutable. `put` is write-once. Nothing else moves.

```rust
store.put_with(key, vec![
    PutSlot::new(SlotSpec::new(len, Placement::Each(vec![cpu0, cpu1])), &src),
]).retention(Retention::Explicit).await?;

store.get(key, SlotIdx(0), &dst_on_gpu3).await?;   // served from cpu:1
store.publish(key, SlotIdx(0), &dst_on_gpu3).await?; // gpu:3 is now a source
```

---

## Proof

One 4 GiB shard, produced on GPU 0, `put` once. Four GB300 GPUs read it.
Nothing changes between the three runs except placement.

| | placement | aggregate | per GPU | served from |
|---|---|---:|---:|---|
| A | `Each([cpu:0, cpu:1])` | **693 GB/s** | 173–185 GB/s | own socket's DRAM |
| B | `Strict(cpu:1)` | 253 GB/s | 63 GB/s (far) / 115 GB/s (near) | one socket, for everyone |
| C | A, then GPU 0 `publish` | **794 GB/s** | 265–310 GB/s | GPU 0 over NVLink; DRAM idle |

Same key. Same bytes. Same code. 3× apart.

Full output: [`docs/weights_fanout.txt`](docs/weights_fanout.txt).

```sh
cargo run --release -p pegastore-cuda --example weights_fanout -- --gib 1 --slots 4
```

---

## What this is for

**Weights.** Put once per node; every rank pulls its slot from its own socket,
then from its NVLink neighbour. Version switches are your control plane's ref,
not the store's problem.

**KV cache.** Checkpoint-style: one object per prefix, one slot per rank.
Partial residency is normal; `Evicted` is an answer, not an exception.

**Hidden states.** `Retention::Explicit` + a full pool = backpressure.
Trainer does `get`, then `remove`.

---

## What this is not

Not a parameter server — nothing is mutable.
Not durable — every tier, SSD included, is a cache.
No leases, no pins, no migration, no quotas, no prefix matching.
The store never owns your GPU memory; it borrows it through `publish`.

Non-goals are enforced at review, not documented as roadmap.

---

## Shape

Two layers, in the OpenDAL tradition:

- `Store` — what you call. Cloneable, layered (`retry`, `metrics`, `timeout`).
- `raw::Access` — what a backend implements. Seven operations.

Backends: `Memory` (the semantic oracle; every other backend must match it),
`Local` (per-NUMA pinned DRAM + CUDA copy engines + NVLink peers),
`Remote` (metaserver + RDMA — next).

Errors are for programs: a closed `ErrorKind`, `Temporary` vs `Permanent`,
context for humans. `Unavailable` is never a miss.

Design notes: [`docs/INTERFACE.md`](docs/INTERFACE.md).

---

Apache-2.0. Parts of the CUDA backend derive from
[pegaflow](https://github.com/pegaflow/pegaflow).
