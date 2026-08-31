//! Operation arguments and replies. `Op*` are plain data: new requirements
//! add fields here, never methods to `Access`.

use std::fmt;

use crate::error::{Error, ErrorKind, Result};
use crate::region::Iov;
use crate::types::{Device, Location, ObjectSpec, Retention, SlotIdx, SlotSpec};

/// Check a gather/scatter list: non-empty, every segment in bounds, all on one
/// device, and (if given) summing to `expect_len`. Returns (device, total).
pub fn validate_iovs(iovs: &[Iov<'_>], expect_len: Option<u64>) -> Result<(Device, u64)> {
    let Some(first) = iovs.first() else {
        return Err(Error::new(ErrorKind::InvalidInput, "empty iov list"));
    };
    let device = first.device();
    let mut total: u64 = 0;
    for (i, iov) in iovs.iter().enumerate() {
        if !iov.in_bounds() {
            return Err(Error::new(ErrorKind::InvalidInput, "iov out of region bounds")
                .with_context("index", i)
                .with_context("offset", iov.offset)
                .with_context("len", iov.len)
                .with_context("region_len", iov.region.len()));
        }
        if iov.device() != device {
            return Err(
                Error::new(ErrorKind::InvalidInput, "iov list spans multiple devices")
                    .with_context("index", i)
                    .with_context("device", iov.device())
                    .with_context("expected", device),
            );
        }
        total = total
            .checked_add(iov.len)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "iov total length overflows"))?;
    }
    if let Some(expect) = expect_len
        && total != expect
    {
        return Err(Error::new(ErrorKind::InvalidInput, "iov total length mismatch")
            .with_context("total", total)
            .with_context("expected", expect));
    }
    Ok((device, total))
}

pub struct PutSlot<'a> {
    pub spec: SlotSpec,
    pub src: &'a [Iov<'a>],
}

impl<'a> PutSlot<'a> {
    pub fn new(spec: SlotSpec, src: &'a [Iov<'a>]) -> Self {
        Self { spec, src }
    }
}

impl fmt::Debug for PutSlot<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PutSlot")
            .field("spec", &self.spec)
            .field("src_segments", &self.src.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct OpPut<'a> {
    pub retention: Retention,
    pub slots: Vec<PutSlot<'a>>,
}

impl<'a> OpPut<'a> {
    pub fn new(slots: Vec<PutSlot<'a>>) -> Self {
        Self {
            retention: Retention::Cache,
            slots,
        }
    }

    pub fn with_retention(mut self, retention: Retention) -> Self {
        self.retention = retention;
        self
    }

    pub fn spec(&self) -> ObjectSpec {
        ObjectSpec {
            retention: self.retention,
            slots: self.slots.iter().map(|s| s.spec.clone()).collect(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.slots.is_empty() {
            return Err(Error::new(ErrorKind::InvalidInput, "object needs at least one slot"));
        }
        for (i, slot) in self.slots.iter().enumerate() {
            validate_iovs(slot.src, Some(slot.spec.len))
                .map_err(|e| e.with_context("slot", i))?;
        }
        Ok(())
    }
}

/// Per-slot outcome of a `put`. The object-level `Result` is reserved for
/// failures that apply to the whole request (`SpecMismatch`, `Unavailable`, ...).
#[derive(Debug)]
pub struct RpPut {
    pub slots: Vec<Result<()>>,
}

impl RpPut {
    pub fn all_ok(&self) -> bool {
        self.slots.iter().all(Result::is_ok)
    }

    /// Collapse to the first per-slot error, if any.
    pub fn into_result(self) -> Result<()> {
        self.slots.into_iter().collect()
    }
}

pub struct OpGet<'a> {
    pub slot: SlotIdx,
    pub src_offset: u64,
    pub dst: &'a [Iov<'a>],
}

impl<'a> OpGet<'a> {
    pub fn new(slot: SlotIdx, dst: &'a [Iov<'a>]) -> Self {
        Self {
            slot,
            src_offset: 0,
            dst,
        }
    }

    pub fn with_src_offset(mut self, src_offset: u64) -> Self {
        self.src_offset = src_offset;
        self
    }

    /// Returns the destination device and total length.
    pub fn validate(&self) -> Result<(Device, u64)> {
        validate_iovs(self.dst, None)
    }
}

impl fmt::Debug for OpGet<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpGet")
            .field("slot", &self.slot)
            .field("src_offset", &self.src_offset)
            .field("dst_segments", &self.dst.len())
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RpGet {
    /// Which replica served the read.
    pub from: Location,
}

pub struct OpPublish<'a> {
    pub slot: SlotIdx,
    pub src: &'a [Iov<'a>],
}

impl<'a> OpPublish<'a> {
    pub fn new(slot: SlotIdx, src: &'a [Iov<'a>]) -> Self {
        Self { slot, src }
    }

    pub fn validate(&self) -> Result<(Device, u64)> {
        validate_iovs(self.src, None)
    }
}

impl fmt::Debug for OpPublish<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpPublish")
            .field("slot", &self.slot)
            .field("src_segments", &self.src.len())
            .finish()
    }
}
