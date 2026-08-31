//! Middleware. The core stays bare; retry, metrics, tracing, timeouts and
//! concurrency limits are layers wrapping a `Servicer`.

use crate::access::Servicer;

pub trait Layer: Send + Sync + 'static {
    fn layer(&self, inner: Servicer) -> Servicer;
}
