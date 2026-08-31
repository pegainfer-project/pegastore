//! Behavior tests: the contract every backend must satisfy. Run against
//! `Memory` here; `Local` / `Remote` reuse the same cases.

use futures::StreamExt;
use pegastore::services::Memory;
use pegastore::{
    Capability, Device, ErrorKind, Key, MemoryRegion, Placement, PutSlot, Retention, SlotIdx,
    SlotSpec, Store, Tier,
};

const CPU0: Device = Device::cpu(0);
const CPU1: Device = Device::cpu(1);
const GPU0: Device = Device::gpu(0);

/// Host buffer + its registration, kept together so the region cannot
/// outlive the bytes.
struct Buf {
    bytes: Vec<u8>,
    region: Option<MemoryRegion>,
}

impl Buf {
    fn filled(store: &Store, device: Device, len: usize, seed: u8) -> Self {
        let bytes: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect();
        Self::from_vec(store, device, bytes)
    }

    fn zeroed(store: &Store, device: Device, len: usize) -> Self {
        Self::from_vec(store, device, vec![0; len])
    }

    fn from_vec(store: &Store, device: Device, mut bytes: Vec<u8>) -> Self {
        // SAFETY: `bytes` lives as long as the region (both are fields of Buf).
        let region = unsafe { store.register(bytes.as_mut_ptr(), bytes.len() as u64, device) }
            .expect("register");
        Self {
            bytes,
            region: Some(region),
        }
    }

    fn region(&self) -> &MemoryRegion {
        self.region.as_ref().unwrap()
    }

    fn iov(&self) -> [pegastore::Iov<'_>; 1] {
        [self.region().iov_all()]
    }

    fn drop_region(&mut self) {
        self.region = None;
    }
}

fn two_numa() -> Store {
    Store::new(
        Memory::builder()
            .device(CPU0)
            .device(CPU1)
            .device(GPU0)
            .gpu_affinity(0, 0)
            .build(),
    )
}

fn slot(len: u64, placement: Placement) -> SlotSpec {
    SlotSpec::new(len, placement)
}

#[tokio::test]
async fn put_get_roundtrip_multi_slot() {
    let store = two_numa();
    let a = Buf::filled(&store, CPU0, 4096, 1);
    let b = Buf::filled(&store, CPU0, 1024, 2);
    let key = Key::from("ns/obj1");

    let rp = store
        .put(
            key.clone(),
            vec![
                PutSlot::new(slot(4096, Placement::Prefer(CPU0)), &a.iov()),
                PutSlot::new(slot(1024, Placement::Prefer(CPU1)), &b.iov()),
            ],
        )
        .await
        .unwrap();
    assert!(rp.all_ok(), "{rp:?}");

    let info = store.stat(std::slice::from_ref(&key)).await.unwrap().remove(0).unwrap();
    assert_eq!(info.retention, Retention::Cache);
    assert_eq!(info.slots.len(), 2);
    assert!(info.is_fully_resident());
    assert_eq!(info.slots[0].replicas[0].location.device, CPU0);
    assert_eq!(info.slots[1].replicas[0].location.device, CPU1);
    assert!(!info.slots[0].replicas[0].misplaced);

    let out = Buf::zeroed(&store, CPU1, 1024);
    let rp = store.get(key.clone(), SlotIdx(1), &out.iov()).await.unwrap();
    assert_eq!(rp.from.device, CPU1);
    assert_eq!(rp.from.tier, Tier::Dram);
    assert_eq!(out.bytes, b.bytes);

    // Ranged read of slot 0 via the builder form.
    let part = Buf::zeroed(&store, CPU0, 100);
    store
        .get_with(key.clone(), SlotIdx(0), &part.iov())
        .src_offset(1000)
        .await
        .unwrap();
    assert_eq!(part.bytes, a.bytes[1000..1100]);
}

#[tokio::test]
async fn write_once_and_spec_mismatch() {
    let store = two_numa();
    let a = Buf::filled(&store, CPU0, 256, 1);
    let key = Key::from("k");

    let rp = store
        .put(key.clone(), vec![PutSlot::new(slot(256, Placement::Anywhere), &a.iov())])
        .await
        .unwrap();
    assert!(rp.all_ok());

    // Same spec again: per-slot AlreadyExists, not an object-level error.
    let rp = store
        .put(key.clone(), vec![PutSlot::new(slot(256, Placement::Anywhere), &a.iov())])
        .await
        .unwrap();
    assert_eq!(rp.slots[0].as_ref().unwrap_err().kind(), ErrorKind::AlreadyExists);

    // Different spec: object-level SpecMismatch.
    let err = store
        .put(key.clone(), vec![PutSlot::new(slot(256, Placement::Strict(CPU0)), &a.iov())])
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::SpecMismatch);

    // Different retention is also a spec mismatch.
    let err = store
        .put_with(key.clone(), vec![PutSlot::new(slot(256, Placement::Anywhere), &a.iov())])
        .retention(Retention::Explicit)
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::SpecMismatch);
}

#[tokio::test]
async fn partial_put_then_repair() {
    // CPU0 holds 1000 bytes: slot 0 (600) fits, slot 1 (600) does not.
    let store = Store::new(Memory::builder().device_with_capacity(CPU0, 1000).build());
    let a = Buf::filled(&store, CPU0, 600, 1);
    let b = Buf::filled(&store, CPU0, 600, 2);
    let key = Key::from("partial");
    let (a_iov, b_iov) = (a.iov(), b.iov());
    let slots = || {
        vec![
            PutSlot::new(slot(600, Placement::Strict(CPU0)), &a_iov),
            PutSlot::new(slot(600, Placement::Strict(CPU0)), &b_iov),
        ]
    };

    let rp = store.put_with(key.clone(), slots()).retention(Retention::Explicit).await.unwrap();
    assert!(rp.slots[0].is_ok());
    assert_eq!(rp.slots[1].as_ref().unwrap_err().kind(), ErrorKind::NoSpace);

    let info = store.stat(std::slice::from_ref(&key)).await.unwrap().remove(0).unwrap();
    assert!(info.slots[0].is_resident());
    assert!(!info.slots[1].is_resident());

    let out = Buf::zeroed(&store, CPU0, 600);
    let err = store.get(key.clone(), SlotIdx(1), &out.iov()).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Evicted);

    // Free space, put again: slot 0 AlreadyExists, slot 1 now lands.
    // (Explicit bytes only leave via remove, so remove the object entirely.)
    store.remove(std::slice::from_ref(&key)).await.unwrap();
    let rp = store.put_with(key.clone(), slots()).retention(Retention::Explicit).await.unwrap();
    assert!(rp.slots[0].is_ok());
    assert_eq!(rp.slots[1].as_ref().unwrap_err().kind(), ErrorKind::NoSpace);

    // Repair path proper: a Cache object elsewhere gets evicted to make room.
    let store = Store::new(Memory::builder().device_with_capacity(CPU0, 1300).build());
    let a = Buf::filled(&store, CPU0, 600, 1);
    let b = Buf::filled(&store, CPU0, 600, 2);
    let filler = Buf::filled(&store, CPU0, 500, 9);
    store
        .put(Key::from("filler"), vec![PutSlot::new(slot(500, Placement::Strict(CPU0)), &filler.iov())])
        .await
        .unwrap()
        .into_result()
        .unwrap();
    let key = Key::from("repair");
    let (a_iov, b_iov) = (a.iov(), b.iov());
    let slots = || {
        vec![
            PutSlot::new(slot(600, Placement::Strict(CPU0)), &a_iov),
            PutSlot::new(slot(600, Placement::Strict(CPU0)), &b_iov),
        ]
    };
    // 500 + 600 = 1100 fits; the second 600 evicts "filler" (Cache) → both land.
    let rp = store.put_with(key.clone(), slots()).retention(Retention::Explicit).await.unwrap();
    assert!(rp.all_ok(), "{rp:?}");
    assert!(store.stat(&[Key::from("filler")]).await.unwrap()[0].is_none());
}

#[tokio::test]
async fn cache_evicts_fifo_explicit_never() {
    let store = Store::new(Memory::builder().device_with_capacity(CPU0, 1000).build());
    let buf = Buf::filled(&store, CPU0, 400, 1);
    let buf_iov = buf.iov();
    let mk = || PutSlot::new(slot(400, Placement::Strict(CPU0)), &buf_iov);

    store.put(Key::from("c1"), vec![mk()]).await.unwrap().into_result().unwrap();
    store.put(Key::from("c2"), vec![mk()]).await.unwrap().into_result().unwrap();
    // Third 400 exceeds 1000 → c1 evicted (FIFO).
    store.put(Key::from("c3"), vec![mk()]).await.unwrap().into_result().unwrap();

    let infos = store
        .stat(&[Key::from("c1"), Key::from("c2"), Key::from("c3")])
        .await
        .unwrap();
    assert!(infos[0].is_none(), "c1 should be gone");
    assert!(infos[1].as_ref().unwrap().is_fully_resident());
    assert!(infos[2].as_ref().unwrap().is_fully_resident());

    // Explicit objects fill the pool; a further Cache put gets NoSpace.
    let store = Store::new(Memory::builder().device_with_capacity(CPU0, 1000).build());
    let buf = Buf::filled(&store, CPU0, 400, 1);
    for k in ["e1", "e2"] {
        store
            .put_with(Key::from(k), vec![PutSlot::new(slot(400, Placement::Strict(CPU0)), &buf.iov())])
            .retention(Retention::Explicit)
            .await
            .unwrap()
            .into_result()
            .unwrap();
    }
    let rp = store
        .put(Key::from("c"), vec![PutSlot::new(slot(400, Placement::Strict(CPU0)), &buf.iov())])
        .await
        .unwrap();
    let err = rp.slots[0].as_ref().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NoSpace);
    assert!(err.is_temporary(), "NoSpace is backpressure, hence retryable");
    // Nothing Explicit was evicted.
    let infos = store.stat(&[Key::from("e1"), Key::from("e2")]).await.unwrap();
    assert!(infos.iter().all(|i| i.as_ref().unwrap().is_fully_resident()));
}

#[tokio::test]
async fn placement_each_and_distance_selection() {
    let store = two_numa();
    let a = Buf::filled(&store, CPU0, 2048, 3);
    let key = Key::from("weights/layer0");
    store
        .put_with(key.clone(), vec![PutSlot::new(slot(2048, Placement::Each(vec![CPU0, CPU1])), &a.iov())])
        .retention(Retention::Explicit)
        .await
        .unwrap()
        .into_result()
        .unwrap();

    let info = store.stat(std::slice::from_ref(&key)).await.unwrap().remove(0).unwrap();
    let devs: Vec<Device> = info.slots[0].replicas.iter().map(|r| r.location.device).collect();
    assert_eq!(devs, vec![CPU0, CPU1]);

    // Reader on CPU1 is served by the CPU1 replica; reader on GPU0 (numa 0) by CPU0.
    let out1 = Buf::zeroed(&store, CPU1, 2048);
    assert_eq!(store.get(key.clone(), SlotIdx(0), &out1.iov()).await.unwrap().from.device, CPU1);
    let out0 = Buf::zeroed(&store, GPU0, 2048);
    assert_eq!(store.get(key.clone(), SlotIdx(0), &out0.iov()).await.unwrap().from.device, CPU0);
    assert_eq!(out0.bytes, a.bytes);
}

#[tokio::test]
async fn prefer_spills_and_marks_misplaced() {
    let store = Store::new(
        Memory::builder()
            .device_with_capacity(CPU0, 100)
            .device(CPU1)
            .build(),
    );
    let a = Buf::filled(&store, CPU0, 500, 4);
    let key = Key::from("spill");
    store
        .put(key.clone(), vec![PutSlot::new(slot(500, Placement::Prefer(CPU0)), &a.iov())])
        .await
        .unwrap()
        .into_result()
        .unwrap();
    let info = store.stat(std::slice::from_ref(&key)).await.unwrap().remove(0).unwrap();
    let r = &info.slots[0].replicas[0];
    assert_eq!(r.location.device, CPU1);
    assert!(r.misplaced);

    // Strict would have refused.
    let err = store
        .put(Key::from("strict"), vec![PutSlot::new(slot(500, Placement::Strict(CPU0)), &a.iov())])
        .await
        .unwrap()
        .into_result()
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NoSpace);
}

#[tokio::test]
async fn publish_external_replica_lives_with_region() {
    let store = two_numa();
    let a = Buf::filled(&store, CPU1, 1024, 5);
    let key = Key::from("w");
    store
        .put(key.clone(), vec![PutSlot::new(slot(1024, Placement::Strict(CPU1)), &a.iov())])
        .await
        .unwrap()
        .into_result()
        .unwrap();

    // A GPU-side consumer fetches, then publishes its copy.
    let mut gpu_copy = Buf::zeroed(&store, GPU0, 1024);
    store.get(key.clone(), SlotIdx(0), &gpu_copy.iov()).await.unwrap();
    store.publish(key.clone(), SlotIdx(0), &gpu_copy.iov()).await.unwrap();

    let info = store.stat(std::slice::from_ref(&key)).await.unwrap().remove(0).unwrap();
    assert_eq!(info.slots[0].replicas.len(), 2);
    assert!(info.slots[0].replicas.iter().any(|r| r.location.tier == Tier::External && r.location.device == GPU0));

    // Another GPU reader is now served from the GPU replica (distance 0), not DRAM.
    let out = Buf::zeroed(&store, GPU0, 1024);
    let rp = store.get(key.clone(), SlotIdx(0), &out.iov()).await.unwrap();
    assert_eq!(rp.from.tier, Tier::External);
    assert_eq!(out.bytes, a.bytes);

    // Length mismatch is rejected.
    let short = Buf::zeroed(&store, GPU0, 512);
    let err = store.publish(key.clone(), SlotIdx(0), &short.iov()).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    // Dropping the region retires the external replica.
    gpu_copy.drop_region();
    let info = store.stat(std::slice::from_ref(&key)).await.unwrap().remove(0).unwrap();
    assert_eq!(info.slots[0].replicas.len(), 1);
    assert_eq!(info.slots[0].replicas[0].location.tier, Tier::Dram);
}

#[tokio::test]
async fn remove_and_remove_prefix() {
    let store = two_numa();
    let a = Buf::filled(&store, CPU0, 64, 6);
    for k in ["m1/x", "m1/y", "m2/x", "m10/x"] {
        store
            .put(Key::from(k), vec![PutSlot::new(slot(64, Placement::Anywhere), &a.iov())])
            .await
            .unwrap()
            .into_result()
            .unwrap();
    }
    assert_eq!(store.remove_prefix(b"m1/").await.unwrap(), 2);
    let infos = store
        .stat(&[Key::from("m1/x"), Key::from("m2/x"), Key::from("m10/x")])
        .await
        .unwrap();
    assert!(infos[0].is_none());
    assert!(infos[1].is_some());
    assert!(infos[2].is_some(), "prefix match is byte-wise, m10/ is not m1/");

    store.remove(&[Key::from("m2/x")]).await.unwrap();
    let out = Buf::zeroed(&store, CPU0, 64);
    let err = store.get(Key::from("m2/x"), SlotIdx(0), &out.iov()).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
    assert!(!err.is_temporary());
    // Removing again is a no-op.
    store.remove(&[Key::from("m2/x")]).await.unwrap();
}

#[tokio::test]
async fn invalid_input_is_rejected_before_the_backend() {
    let store = two_numa();
    let a = Buf::filled(&store, CPU0, 100, 7);
    let key = Key::from("bad");

    // Declared len != iov total.
    let err = store
        .put(key.clone(), vec![PutSlot::new(slot(200, Placement::Anywhere), &a.iov())])
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    // Iov out of region bounds.
    let oob = [a.region().iov(50, 100)];
    let err = store
        .put(key.clone(), vec![PutSlot::new(slot(100, Placement::Anywhere), &oob)])
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    // Empty iov list.
    let err = store
        .put(key.clone(), vec![PutSlot::new(slot(0, Placement::Anywhere), &[])])
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    store
        .put(key.clone(), vec![PutSlot::new(slot(100, Placement::Anywhere), &a.iov())])
        .await
        .unwrap()
        .into_result()
        .unwrap();

    // Read past the slot.
    let out = Buf::zeroed(&store, CPU0, 60);
    let err = store
        .get_with(key.clone(), SlotIdx(0), &out.iov())
        .src_offset(50)
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    // Unknown slot.
    let err = store.get(key.clone(), SlotIdx(7), &out.iov()).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[tokio::test]
async fn capability_gates_unsupported_features() {
    let cap = Capability {
        publish: false,
        placement_each: false,
        retention_explicit: false,
        remove_prefix: false,
        gpu_memory: false,
        max_iov: Some(1),
        ..Capability::all()
    };
    let store = Store::new(Memory::builder().device(CPU0).capability(cap).build());
    let a = Buf::filled(&store, CPU0, 32, 8);

    // SAFETY: rejected by the capability check before any pointer use.
    let err = unsafe { store.register(std::ptr::null_mut(), 0, GPU0) }.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(
        store
            .put(Key::from("k"), vec![PutSlot::new(slot(32, Placement::Each(vec![CPU0])), &a.iov())])
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        store
            .put_with(Key::from("k"), vec![PutSlot::new(slot(32, Placement::Anywhere), &a.iov())])
            .retention(Retention::Explicit)
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(store.remove_prefix(b"k").await.unwrap_err().kind(), ErrorKind::Unsupported);
    assert_eq!(
        store.publish(Key::from("k"), SlotIdx(0), &a.iov()).await.unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    let two = [a.region().iov(0, 16), a.region().iov(16, 16)];
    assert_eq!(
        store
            .put(Key::from("k"), vec![PutSlot::new(slot(32, Placement::Anywhere), &two)])
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
}

#[tokio::test]
async fn get_many_streams_in_completion_order() {
    let store = two_numa();
    let bufs: Vec<Buf> = (0..8).map(|i| Buf::filled(&store, CPU0, 512, i as u8)).collect();
    let key = Key::from("layers");
    let iovs: Vec<[pegastore::Iov<'_>; 1]> = bufs.iter().map(|b| b.iov()).collect();
    let slots: Vec<PutSlot<'_>> = iovs
        .iter()
        .map(|iov| PutSlot::new(slot(512, Placement::Anywhere), iov))
        .collect();
    store.put(key.clone(), slots).await.unwrap().into_result().unwrap();

    let outs: Vec<Buf> = (0..8).map(|_| Buf::zeroed(&store, CPU1, 512)).collect();
    let out_iovs: Vec<[pegastore::Iov<'_>; 1]> = outs.iter().map(|o| o.iov()).collect();
    let reqs = out_iovs
        .iter()
        .enumerate()
        .map(|(i, iov)| (key.clone(), pegastore::OpGet::new(SlotIdx(i as u32), iov)))
        .collect();

    let mut seen = [false; 8];
    let mut stream = store.get_many(reqs);
    while let Some((i, r)) = stream.next().await {
        r.unwrap();
        seen[i] = true;
    }
    assert!(seen.iter().all(|s| *s));
    for (o, b) in outs.iter().zip(&bufs) {
        assert_eq!(o.bytes, b.bytes);
    }
}
