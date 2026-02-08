use std::fmt;

use crate::TensorShape;

#[derive(Debug)]
pub enum Error {
    LayoutMismatch {
        a: TensorShape,
        b: TensorShape,
    },
    RankMismatch {
        expected: usize,
        got:      usize,
    },
    AxisOutOfBounds {
        axis: usize,
        rank: usize,
    },
    DuplicateAxis {
        axis: usize,
    },
    IndexOutOfBounds {
        axis:  usize,
        index: usize,
        dim:   usize,
    },
    LinearIndexOutOfRange {
        index: usize,
        size:  usize,
    },
    InvalidMergeRange {
        start: usize,
        end:   usize,
        rank:  usize,
    },
    TooManySplitWildcards,
    InvalidDimensionSplit {
        original_size: usize,
        shape:         Vec<usize>,
    },
    TensorSizeMismatch {
        expected: Vec<usize>,
        got:      Vec<usize>,
    },
    InvalidSkipStep {
        step: usize,
    },
    MatMulInvalidShape {
        a_shape: TensorShape,
        b_shape: TensorShape,
    },
    MatMulDimensionMismatch {
        a_last:        usize,
        b_second_last: usize,
        a_shape:       TensorShape,
        b_shape:       TensorShape,
    },
    DTypeCastFailed {
        from: &'static str,
        to:   &'static str,
    },
    RequiresGradUnsupported {
        op: &'static str,
    },
    UnsupportedOperation {
        op:      &'static str,
        backend: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::LayoutMismatch { a, b } => {
                write!(
                    f,
                    "layout mismatch:\n shape a:\n{:?}\n shape b:\n{:?}",
                    a, b
                )
            }
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
            Error::TooManySplitWildcards => {
                write!(f, "Cannot have more than one wildcard (0) in split sizes")
            }
            Error::InvalidDimensionSplit { original_size, shape } => {
                write!(
                    f,
                    "Cannot split dimension of size {} into sizes {:?}",
                    original_size, shape
                )
            }
            Error::TensorSizeMismatch { expected, got } => {
                write!(f, "expected tensor size {:?}, got {:?}", expected, got)
            }
            Error::InvalidSkipStep { step } => {
                write!(f, "invalid skip step of {}", step)
            }
            Error::MatMulInvalidShape { a_shape, b_shape } => {
                // FIXME
                write!(
                    f,
                    "matmul invalid shape: a shape {:?}, b shape {:?} (both must have rank >= 2)",
                    a_shape, b_shape
                )
            }
            Error::MatMulDimensionMismatch {
                a_last,
                b_second_last,
                a_shape,
                b_shape,
            } => {
                write!(
                    f,
                    "matmul dimension mismatch: a last dimension {} (shape {:?}) does not match b \
                     second last dimension {} (shape {:?})",
                    a_last, a_shape, b_second_last, b_shape
                )
            }
            Error::DTypeCastFailed { from, to } => {
                write!(f, "dtype cast failed: from {} to {}", from, to)
            }
            Error::RequiresGradUnsupported { op } => {
                write!(
                    f,
                    "operation {} is not supported when requires_grad = true",
                    op
                )
            }
            Error::UnsupportedOperation { op, backend } => {
                write!(
                    f,
                    "operation {} is not supported on backend {}",
                    op, backend
                )
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
