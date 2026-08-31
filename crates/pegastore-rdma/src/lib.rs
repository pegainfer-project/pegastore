// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RDMA transfer engine for pegastore (RC verbs via `sideway`), derived from
//! `pegaflow-transfer` v1.
//!
//! Model:
//! - A [`TransferEngine`] drives a fixed list of NICs. Two engines connect
//!   with an out-of-band handshake that carries QP endpoints only.
//! - Memory is registered independently of any connection. Registration
//!   returns a serializable [`RegionDescriptor`] (address range + one rkey
//!   per NIC); whoever holds it can READ/WRITE the memory once connected.
//!   Host memory and dma-buf-exported device memory both register.
//! - Each op in a batch goes out on a NIC local to its registered memory's
//!   NUMA node, so a value spread over both sockets uses both sockets' NICs.

mod engine;
mod error;
pub mod numa;
mod rc_backend;
pub mod rdma_topo;

pub use engine::{
    ConnectionStatus, HandshakeMetadata, RegionDescriptor, TransferDesc, TransferEngine,
    TransferOp,
};
pub use error::{Result, TransferError};
pub use numa::NumaNode;

/// Minimal stderr logger so the engine's `log::` calls are visible without a
/// logging framework. Idempotent.
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
