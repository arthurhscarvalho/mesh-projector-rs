#[macro_export]
macro_rules! conditional_log {
    ($condition:expr, $level:ident, $($arg:tt)+) => {
        if $condition {
            log::$level!($($arg)+);
        }
    };
}

#[macro_export]
macro_rules! log_removed_points {
    ($logging_enabled:expr, $freqs:expr) => {
        if $logging_enabled {
            for i in 0..6 {
                log::info!("{} points removed from projection {}", $freqs[i], i);
            }
        }
    };
}
