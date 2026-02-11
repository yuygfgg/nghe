mod format;
mod sink;
mod transcoder;

pub use sink::Sink;
pub use transcoder::Transcoder;
use typed_path::Utf8PlatformPathBuf;

#[derive(Debug)]
pub struct Path {
    pub input: String,
    pub output: Option<Utf8PlatformPathBuf>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Trim {
    /// Start position in seconds.
    pub start: f64,
    /// Optional duration in seconds.
    pub duration: Option<f64>,
}

impl Trim {
    pub fn from_offset(offset_seconds: u32) -> Self {
        Self { start: offset_seconds as f64, duration: None }
    }
}
