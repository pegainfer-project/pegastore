//! The backend trait. `Access` is what implementers write (RPITIT, zero
//! boxing); `AccessDyn` is its object-safe mirror used by `Store` and layers.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::ops::{OpGet, OpPublish, OpPut, RpGet, RpPut};
use crate::region::MemoryRegion;
use crate::types::{AccessInfo, Device, Key, ObjectInfo};

pub type BoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub trait Access: Send + Sync + 'static {
    fn info(&self) -> Arc<AccessInfo>;

    /// Register user memory.
    ///
    /// # Safety
    /// `[ptr, ptr + len)` must be valid, accessible according to `device`,
    /// and outlive the returned region.
    unsafe fn register(&self, ptr: *mut u8, len: u64, device: Device) -> Result<MemoryRegion>;

    /// Write-once per slot. Creates the object if absent; otherwise the spec
    /// must match and only empty slots are written.
    fn put<'a>(&'a self, key: Key, op: OpPut<'a>) -> impl Future<Output = Result<RpPut>> + Send + 'a;

    /// Register user memory as an `External` replica of an existing slot.
    fn publish<'a>(
        &'a self,
        key: Key,
        op: OpPublish<'a>,
    ) -> impl Future<Output = Result<()>> + Send + 'a;

    fn stat<'a>(
        &'a self,
        keys: &'a [Key],
    ) -> impl Future<Output = Result<Vec<Option<ObjectInfo>>>> + Send + 'a;

    /// Read one slot into `dst`, choosing the nearest replica.
    fn get<'a>(&'a self, key: Key, op: OpGet<'a>) -> impl Future<Output = Result<RpGet>> + Send + 'a;

    fn remove<'a>(&'a self, keys: &'a [Key]) -> impl Future<Output = Result<()>> + Send + 'a;

    /// Remove every object whose key starts with `prefix`; returns the count.
    fn remove_prefix<'a>(&'a self, prefix: &'a [u8]) -> impl Future<Output = Result<u64>> + Send + 'a;
}

pub trait AccessDyn: Send + Sync + 'static {
    fn info_dyn(&self) -> Arc<AccessInfo>;
    /// # Safety
    /// See [`Access::register`].
    unsafe fn register_dyn(&self, ptr: *mut u8, len: u64, device: Device) -> Result<MemoryRegion>;
    fn put_dyn<'a>(&'a self, key: Key, op: OpPut<'a>) -> BoxedFuture<'a, Result<RpPut>>;
    fn publish_dyn<'a>(&'a self, key: Key, op: OpPublish<'a>) -> BoxedFuture<'a, Result<()>>;
    fn stat_dyn<'a>(&'a self, keys: &'a [Key]) -> BoxedFuture<'a, Result<Vec<Option<ObjectInfo>>>>;
    fn get_dyn<'a>(&'a self, key: Key, op: OpGet<'a>) -> BoxedFuture<'a, Result<RpGet>>;
    fn remove_dyn<'a>(&'a self, keys: &'a [Key]) -> BoxedFuture<'a, Result<()>>;
    fn remove_prefix_dyn<'a>(&'a self, prefix: &'a [u8]) -> BoxedFuture<'a, Result<u64>>;
}

impl<T: Access> AccessDyn for T {
    fn info_dyn(&self) -> Arc<AccessInfo> {
        self.info()
    }

    unsafe fn register_dyn(&self, ptr: *mut u8, len: u64, device: Device) -> Result<MemoryRegion> {
        // SAFETY: forwarded contract.
        unsafe { self.register(ptr, len, device) }
    }

    fn put_dyn<'a>(&'a self, key: Key, op: OpPut<'a>) -> BoxedFuture<'a, Result<RpPut>> {
        Box::pin(self.put(key, op))
    }

    fn publish_dyn<'a>(&'a self, key: Key, op: OpPublish<'a>) -> BoxedFuture<'a, Result<()>> {
        Box::pin(self.publish(key, op))
    }

    fn stat_dyn<'a>(&'a self, keys: &'a [Key]) -> BoxedFuture<'a, Result<Vec<Option<ObjectInfo>>>> {
        Box::pin(self.stat(keys))
    }

    fn get_dyn<'a>(&'a self, key: Key, op: OpGet<'a>) -> BoxedFuture<'a, Result<RpGet>> {
        Box::pin(self.get(key, op))
    }

    fn remove_dyn<'a>(&'a self, keys: &'a [Key]) -> BoxedFuture<'a, Result<()>> {
        Box::pin(self.remove(keys))
    }

    fn remove_prefix_dyn<'a>(&'a self, prefix: &'a [u8]) -> BoxedFuture<'a, Result<u64>> {
        Box::pin(self.remove_prefix(prefix))
    }
}

/// The type `Store` and layers hold.
pub type Servicer = Arc<dyn AccessDyn>;

impl<T: AccessDyn + ?Sized> Access for Arc<T> {
    fn info(&self) -> Arc<AccessInfo> {
        self.as_ref().info_dyn()
    }

    unsafe fn register(&self, ptr: *mut u8, len: u64, device: Device) -> Result<MemoryRegion> {
        // SAFETY: forwarded contract.
        unsafe { self.as_ref().register_dyn(ptr, len, device) }
    }

    fn put<'a>(&'a self, key: Key, op: OpPut<'a>) -> impl Future<Output = Result<RpPut>> + Send + 'a {
        self.as_ref().put_dyn(key, op)
    }

    fn publish<'a>(
        &'a self,
        key: Key,
        op: OpPublish<'a>,
    ) -> impl Future<Output = Result<()>> + Send + 'a {
        self.as_ref().publish_dyn(key, op)
    }

    fn stat<'a>(
        &'a self,
        keys: &'a [Key],
    ) -> impl Future<Output = Result<Vec<Option<ObjectInfo>>>> + Send + 'a {
        self.as_ref().stat_dyn(keys)
    }

    fn get<'a>(&'a self, key: Key, op: OpGet<'a>) -> impl Future<Output = Result<RpGet>> + Send + 'a {
        self.as_ref().get_dyn(key, op)
    }

    fn remove<'a>(&'a self, keys: &'a [Key]) -> impl Future<Output = Result<()>> + Send + 'a {
        self.as_ref().remove_dyn(keys)
    }

    fn remove_prefix<'a>(&'a self, prefix: &'a [u8]) -> impl Future<Output = Result<u64>> + Send + 'a {
        self.as_ref().remove_prefix_dyn(prefix)
    }
}
