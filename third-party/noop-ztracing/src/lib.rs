//! Apache-licensed no-op implementation of GPUI's disabled tracing surface.
//!
//! Vecgra does not enable Zed's `ZTRACING` build-time instrumentation. GPUI's
//! disabled configuration only needs span-producing macros, a passthrough
//! `instrument` attribute, and a span value whose methods do nothing.

pub use tracing::{Level, field};
pub use ztracing_macro::instrument;

/// A disabled tracing span.
#[derive(Clone, Copy, Debug, Default)]
pub struct Span;

impl Span {
    /// Returns the disabled current span.
    pub fn current() -> Self {
        Self
    }

    /// Enters the disabled span.
    pub fn enter(&self) {}

    /// Discards a field recording operation.
    pub fn record<T, S>(&self, _field: T, _value: S) {}
}

/// Initializes tracing. Instrumentation is intentionally disabled in Vecgra.
pub fn init() {}

#[doc(hidden)]
#[macro_export]
macro_rules! __disabled_span {
    ($($tokens:tt)*) => {
        $crate::Span
    };
}

pub use __disabled_span as debug_span;
pub use __disabled_span as error_span;
pub use __disabled_span as event;
pub use __disabled_span as info_span;
pub use __disabled_span as span;
pub use __disabled_span as trace_span;
pub use __disabled_span as warn_span;
