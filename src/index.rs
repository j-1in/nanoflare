use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

#[derive(Debug, Clone, Copy)]
pub struct Rest;

#[derive(Debug, Clone, Copy)]
pub enum TensorIndex {
    Single(usize),
    Slice {
        start: Option<usize>,
        end:   Option<usize>,
    },
    Rest,
}

impl From<RangeFull> for TensorIndex {
    fn from(_: RangeFull) -> Self {
        TensorIndex::Slice { start: None, end: None }
    }
}

impl From<Range<usize>> for TensorIndex {
    fn from(range: Range<usize>) -> Self {
        TensorIndex::Slice {
            start: Some(range.start),
            end:   Some(range.end),
        }
    }
}

impl From<RangeFrom<usize>> for TensorIndex {
    fn from(range: RangeFrom<usize>) -> Self {
        TensorIndex::Slice {
            start: Some(range.start),
            end:   None,
        }
    }
}

impl From<RangeTo<usize>> for TensorIndex {
    fn from(range: RangeTo<usize>) -> Self {
        TensorIndex::Slice { start: None, end: Some(range.end) }
    }
}

impl From<RangeInclusive<usize>> for TensorIndex {
    fn from(range: RangeInclusive<usize>) -> Self {
        TensorIndex::Slice {
            start: Some(*range.start()),
            end:   Some(*range.end() + 1),
        }
    }
}

impl From<RangeToInclusive<usize>> for TensorIndex {
    fn from(range: RangeToInclusive<usize>) -> Self {
        TensorIndex::Slice {
            start: None,
            end:   Some(range.end + 1),
        }
    }
}

impl From<usize> for TensorIndex {
    fn from(index: usize) -> Self {
        TensorIndex::Single(index)
    }
}

impl From<Rest> for TensorIndex {
    fn from(_: Rest) -> Self {
        TensorIndex::Rest
    }
}

#[macro_export]
macro_rules! i {
    // Matches patterns like i![1, .., Rest]
    ($($x:expr),* $(,)?) => {
        [
            $(
                $crate::TensorIndex::from($x)
            ),*
        ]
    };
}
