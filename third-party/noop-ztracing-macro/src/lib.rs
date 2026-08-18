//! Passthrough attribute used when GPUI tracing is disabled.

/// Leaves an instrumented item unchanged.
#[proc_macro_attribute]
pub fn instrument(
    _arguments: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    item
}
