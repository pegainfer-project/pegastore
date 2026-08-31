//! User-facing handle. Cheap to clone. Validates requests against the
//! backend's `Capability` before delegating; composes `get_many`.

use std::future::IntoFuture;
use std::sync::Arc;

use futures::stream::{FuturesUnordered, Stream};

use crate::access::{Access, BoxedFuture, Servicer};
use crate::error::{Error, ErrorKind, Result};
use crate::layer::Layer;
use crate::ops::{OpGet, OpPublish, OpPut, PutSlot, RpGet, RpPut};
use crate::region::{Iov, MemoryRegion};
use crate::types::{AccessInfo, Device, Key, ObjectInfo, Placement, Retention, SlotIdx};

#[derive(Clone)]
pub struct Store {
    srv: Servicer,
    info: Arc<AccessInfo>,
}

impl Store {
    pub fn new(access: impl Access) -> Self {
        Self::from_servicer(Arc::new(access))
    }

    pub fn from_servicer(srv: Servicer) -> Self {
        let info = srv.info_dyn();
        Self { srv, info }
    }

    pub fn layer<L: Layer>(self, layer: L) -> Self {
        Self::from_servicer(layer.layer(self.srv))
    }

    pub fn info(&self) -> &AccessInfo {
        &self.info
    }

    pub fn servicer(&self) -> &Servicer {
        &self.srv
    }

    // ---- memory ----

    /// Register caller memory so `Iov`s can point into it.
    ///
    /// # Safety
    /// `[ptr, ptr + len)` must be valid, accessible according to `device`,
    /// and outlive the returned `MemoryRegion`.
    pub unsafe fn register(&self, ptr: *mut u8, len: u64, device: Device) -> Result<MemoryRegion> {
        if device.is_gpu() && !self.info.capability.gpu_memory {
            return Err(unsupported("gpu memory").with_operation("register"));
        }
        // SAFETY: forwarded contract.
        unsafe { self.srv.register_dyn(ptr, len, device) }
    }

    // ---- put ----

    pub async fn put(&self, key: Key, slots: Vec<PutSlot<'_>>) -> Result<RpPut> {
        self.put_options(key, OpPut::new(slots)).await
    }

    pub fn put_with<'a>(&'a self, key: Key, slots: Vec<PutSlot<'a>>) -> FuturePut<'a> {
        FuturePut {
            store: self,
            key,
            op: OpPut::new(slots),
        }
    }

    pub async fn put_options(&self, key: Key, op: OpPut<'_>) -> Result<RpPut> {
        self.check_put(&op).map_err(|e| e.with_operation("put"))?;
        self.srv.put_dyn(key, op).await.map_err(|e| e.with_operation("put"))
    }

    fn check_put(&self, op: &OpPut<'_>) -> Result<()> {
        op.validate()?;
        let cap = &self.info.capability;
        if op.retention == Retention::Explicit && !cap.retention_explicit {
            return Err(unsupported("retention explicit"));
        }
        if let Some(max) = cap.max_slots
            && op.slots.len() as u64 > u64::from(max)
        {
            return Err(unsupported("too many slots").with_context("max_slots", max));
        }
        for (i, slot) in op.slots.iter().enumerate() {
            match &slot.spec.placement {
                Placement::Strict(_) if !cap.placement_strict => {
                    return Err(unsupported("placement strict").with_context("slot", i));
                }
                Placement::Each(_) if !cap.placement_each => {
                    return Err(unsupported("placement each").with_context("slot", i));
                }
                _ => {}
            }
            if let Some(max) = cap.max_slot_len
                && slot.spec.len > max
            {
                return Err(unsupported("slot too large")
                    .with_context("slot", i)
                    .with_context("max_slot_len", max));
            }
            self.check_iov_count(slot.src.len())?;
        }
        Ok(())
    }

    // ---- get ----

    pub async fn get(&self, key: Key, slot: SlotIdx, dst: &[Iov<'_>]) -> Result<RpGet> {
        self.get_options(key, OpGet::new(slot, dst)).await
    }

    pub fn get_with<'a>(&'a self, key: Key, slot: SlotIdx, dst: &'a [Iov<'a>]) -> FutureGet<'a> {
        FutureGet {
            store: self,
            key,
            op: OpGet::new(slot, dst),
        }
    }

    pub async fn get_options(&self, key: Key, op: OpGet<'_>) -> Result<RpGet> {
        op.validate().map_err(|e| e.with_operation("get"))?;
        self.check_iov_count(op.dst.len()).map_err(|e| e.with_operation("get"))?;
        self.srv.get_dyn(key, op).await.map_err(|e| e.with_operation("get"))
    }

    /// Issue many gets concurrently; yields `(index, result)` in completion
    /// order. This is the vehicle for layer-wise overlap.
    pub fn get_many<'a>(
        &'a self,
        reqs: Vec<(Key, OpGet<'a>)>,
    ) -> impl Stream<Item = (usize, Result<RpGet>)> + 'a {
        reqs.into_iter()
            .enumerate()
            .map(|(i, (key, op))| -> BoxedFuture<'a, (usize, Result<RpGet>)> {
                Box::pin(async move { (i, self.get_options(key, op).await) })
            })
            .collect::<FuturesUnordered<_>>()
    }

    // ---- publish ----

    pub async fn publish(&self, key: Key, slot: SlotIdx, src: &[Iov<'_>]) -> Result<()> {
        if !self.info.capability.publish {
            return Err(unsupported("publish").with_operation("publish"));
        }
        let op = OpPublish::new(slot, src);
        op.validate().map_err(|e| e.with_operation("publish"))?;
        self.check_iov_count(src.len()).map_err(|e| e.with_operation("publish"))?;
        self.srv.publish_dyn(key, op).await.map_err(|e| e.with_operation("publish"))
    }

    // ---- metadata / lifecycle ----

    pub async fn stat(&self, keys: &[Key]) -> Result<Vec<Option<ObjectInfo>>> {
        if let Some(max) = self.info.capability.max_stat_batch
            && keys.len() as u64 > u64::from(max)
        {
            return Err(unsupported("stat batch too large")
                .with_context("max_stat_batch", max)
                .with_operation("stat"));
        }
        self.srv.stat_dyn(keys).await.map_err(|e| e.with_operation("stat"))
    }

    pub async fn remove(&self, keys: &[Key]) -> Result<()> {
        self.srv.remove_dyn(keys).await.map_err(|e| e.with_operation("remove"))
    }

    pub async fn remove_prefix(&self, prefix: &[u8]) -> Result<u64> {
        if !self.info.capability.remove_prefix {
            return Err(unsupported("remove_prefix").with_operation("remove_prefix"));
        }
        self.srv
            .remove_prefix_dyn(prefix)
            .await
            .map_err(|e| e.with_operation("remove_prefix"))
    }

    fn check_iov_count(&self, n: usize) -> Result<()> {
        if let Some(max) = self.info.capability.max_iov
            && n as u64 > u64::from(max)
        {
            return Err(unsupported("too many iov segments").with_context("max_iov", max));
        }
        Ok(())
    }
}

fn unsupported(what: &str) -> Error {
    Error::new(ErrorKind::Unsupported, format!("backend does not support {what}"))
}

/// Builder form of `put`: `store.put_with(key, slots).retention(Explicit).await`.
pub struct FuturePut<'a> {
    store: &'a Store,
    key: Key,
    op: OpPut<'a>,
}

impl FuturePut<'_> {
    pub fn retention(mut self, retention: Retention) -> Self {
        self.op.retention = retention;
        self
    }
}

impl<'a> IntoFuture for FuturePut<'a> {
    type Output = Result<RpPut>;
    type IntoFuture = BoxedFuture<'a, Result<RpPut>>;

    fn into_future(self) -> Self::IntoFuture {
        let Self { store, key, op } = self;
        Box::pin(store.put_options(key, op))
    }
}

/// Builder form of `get`: `store.get_with(key, slot, dst).src_offset(n).await`.
pub struct FutureGet<'a> {
    store: &'a Store,
    key: Key,
    op: OpGet<'a>,
}

impl FutureGet<'_> {
    pub fn src_offset(mut self, src_offset: u64) -> Self {
        self.op.src_offset = src_offset;
        self
    }
}

impl<'a> IntoFuture for FutureGet<'a> {
    type Output = Result<RpGet>;
    type IntoFuture = BoxedFuture<'a, Result<RpGet>>;

    fn into_future(self) -> Self::IntoFuture {
        let Self { store, key, op } = self;
        Box::pin(store.get_options(key, op))
    }
}
