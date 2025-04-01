#[macro_export]
macro_rules! conditional_log {
    ($condition:expr, $level:ident, $($arg:tt)+) => {
        if $condition {
            log::$level!($($arg)+);
        }
    };
}

#[macro_export]
macro_rules! range_conditional_log {
    ($logging_enabled:expr, $range:expr, $level:ident, $freqs:expr, $message:expr) => {
        if $logging_enabled {
            for i in $range {
                conditional_log!(true, $level, $message, $freqs[i], i);
            }
        }
    };
}
