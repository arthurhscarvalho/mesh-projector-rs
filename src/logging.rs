use log::LevelFilter;
use std::io;

/// Initialize the logger for the Rust library.
/// This sets up colog for console logging.
pub fn init_logger() -> Result<(), io::Error> {
    colog::init();
    log::set_max_level(LevelFilter::Info);
    Ok(())
}
