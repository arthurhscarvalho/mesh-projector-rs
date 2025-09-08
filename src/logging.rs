use log::LevelFilter;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

static IS_LOGGER_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Initialize the logger for the Rust library.
/// This sets up colog for console logging.
/// This function is not thread-safe and should not be called from multiple threads.
pub fn init_logger() -> Result<(), io::Error> {
    colog::init();
    log::set_max_level(LevelFilter::Info);
    Ok(())
}

/// Initialize the logger for the Rust library.
/// This function is thread-safe and ensures that the logger is initialized only once.
pub fn atomic_init_logger(verbose: bool) -> Result<(), io::Error> {
    if verbose && !IS_LOGGER_INITIALIZED.load(Ordering::Acquire) {
        init_logger()?;
        IS_LOGGER_INITIALIZED.store(true, Ordering::Release);
    }
    Ok(())
}
