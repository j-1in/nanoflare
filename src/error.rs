use std::fmt;

#[derive(Debug)]
pub enum Error {
    RankMismatch { expected: usize, got: usize },
    AxisOutOfBounds { axis: usize, rank: usize },
    DuplicateAxis { axis: usize },
    IndexOutOfBounds { axis: usize, index: usize, dim: usize },
    LinearIndexOutOfRange { index: usize, size: usize },
    InvalidMergeRange { start: usize, end: usize, rank: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::RankMismatch { expected, got } => {
                write!(f, "expected {} axes, got {}", expected, got)
            }
            Error::AxisOutOfBounds { axis, rank } => {
                write!(f, "axis {} out of bounds for tensor of rank {}", axis, rank)
            }
            Error::DuplicateAxis { axis } => {
                write!(f, "duplicate axis {}", axis)
            }
            Error::IndexOutOfBounds { axis, index, dim } => {
                write!(
                    f,
                    "index {} out of bounds for axis {} with dimension size {}",
                    index, axis, dim
                )
            }
            Error::LinearIndexOutOfRange { index, size } => {
                write!(f, "linear index {} out of range for size {}", index, size)
            }
            Error::InvalidMergeRange { start, end, rank } => {
                write!(
                    f,
                    "invalid merge range {}..={} for tensor of rank {}",
                    start, end, rank
                )
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
