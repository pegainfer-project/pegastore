// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RDMA transfer engine, relocated verbatim from `pegaflow-transfer` (v1).
//!
//! This commit is a pure move: only `use` paths changed. Interface changes
//! (memory registration decoupled from the handshake, device-aware MRs)
//! land in follow-up commits.

mod engine;
mod error;
pub mod numa;
mod rc_backend;
pub mod rdma_topo;

pub use engine::{
    ConnectionStatus, HandshakeMetadata, MemoryRegion, TransferDesc, TransferEngine, TransferOp,
};
pub use error::{Result, TransferError};

/// Minimal stderr logger so the relocated code's `log::` calls are visible
/// without pulling pegaflow-common's logging stack. Idempotent.
pub fn init_logging() {
    struct Stderr;
    impl log::Log for Stderr {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= log::Level::Debug
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                eprintln!("[{}] {}: {}", r.level(), r.target(), r.args());
            }
        }
        fn flush(&self) {}
    }
    static LOGGER: Stderr = Stderr;
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Debug);
    }
}
