use std::fmt;

#[derive(Debug)]
pub enum Error {
    RankMismatch { expected: usize, got: usize },
    AxisOutOfBounds { axis: usize, rank: usize },
    DuplicateAxis { axis: usize },
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
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
