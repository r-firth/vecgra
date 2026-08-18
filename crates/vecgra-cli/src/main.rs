mod ann_benchmark;
mod commands;
mod embedder;
mod exporter;
mod github;
mod graph_benchmark;
mod importer;
mod jsonl;
mod query;

fn main() {
    restore_sigpipe_default();
    if let Err(error) = commands::run() {
        eprintln!("error: {error}");
        let mut source = error.source();
        while let Some(cause) = source {
            eprintln!("  caused by: {cause}");
            source = cause.source();
        }
        std::process::exit(1);
    }
}

#[cfg(unix)]
fn restore_sigpipe_default() {
    // SAFETY: signal is called before the CLI starts worker threads, and both
    // constants are supplied by libc for this target. Restoring SIGPIPE's
    // default disposition gives pipelines conventional Unix behavior.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn restore_sigpipe_default() {}
