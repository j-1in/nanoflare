use crate::{Error, Result};
use std::fmt::Debug;
use std::ops::{Index, RangeInclusive, RangeTo};

#[derive(Debug, Clone, PartialEq)]
pub struct TensorLayout {
    shape:   TensorShape,
    strides: Vec<usize>,
}

impl TensorLayout {
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    pub fn new(shape: Vec<usize>) -> Self {
        let strides = Self::compute_stride(&shape);

        TensorLayout {
            shape: TensorShape::new(shape),
            strides,
        }
    }

    pub fn compute_stride(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];

        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }

    /// Convert multi-dimensional indices into a flat (raveled) index.
    ///
    /// Returns an error if the number of indices doesn't match the tensor
    /// rank or if any index is out of bounds for its corresponding axis.
    pub fn ravel_index(&self, indices: &[usize]) -> Result<usize> {
        let rank = self.shape.len();
        if indices.len() != rank {
            return Err(Error::RankMismatch {
                expected: rank,
                got:      indices.len(),
            });
        }

        // Check each index is within bounds of its corresponding dimension.
        for (axis, (&idx, &dim)) in indices.iter().zip(self.shape.as_slice()).enumerate() {
            if idx >= dim {
                return Err(Error::IndexOutOfBounds { axis, index: idx, dim });
            }
        }

        let flat = indices
            .iter()
            .zip(&self.strides)
            .fold(0usize, |acc, (idx, stride)| acc + idx * stride);

        Ok(flat)
    }

    /// Convert a flat (raveled) index into multi-dimensional indices.
    ///
    /// Returns an error if the flat index is out of range for the tensor size.
    pub fn unravel_index(&self, index: usize) -> Result<Vec<usize>> {
        if self.shape.is_empty() {
            return Ok(vec![]);
        }

        let total = self.shape.size();
        if index >= total {
            return Err(Error::LinearIndexOutOfRange { index, size: total });
        }

        let mut indices = vec![0; self.shape.len()];
        let mut remainder = index;

        for i in 0..self.shape.len() {
            indices[i] = remainder / self.strides[i];
            remainder %= self.strides[i];
        }

        Ok(indices)
    }

    /// Permute the dimensions of the tensor layout according to the given
    /// indices.
    ///
    /// # Arguments
    /// * `permuted_indices` - A slice of usize representing the new order of
    ///   dimensions.
    ///
    /// # Returns
    /// * A new `TensorLayout` with permuted shape and strides.
    ///
    /// # Example
    /// ```rust
    /// let layout = TensorLayout::new(vec![2, 3, 4]);
    /// let permuted_layout = layout.permute(&[2, 0, 1]);
    /// assert_eq!(permuted_layout.shape().as_slice(), &[4, 2, 3]);
    /// ```
    ///
    /// # Errors
    /// * Returns an error if `permuted_indices` is not a valid permutation of
    ///   the tensor dimensions.
    pub fn permute(&self, permuted_indices: &[usize]) -> Result<Self> {
        let rank = self.shape.len();

        // 1. Check rank
        if permuted_indices.len() != rank {
            return Err(Error::RankMismatch {
                expected: rank,
                got:      permuted_indices.len(),
            });
        }

        // 2. Check bounds + duplicates
        let mut seen = vec![false; rank];
        for &axis in permuted_indices {
            if axis >= rank {
                return Err(Error::AxisOutOfBounds { axis, rank });
            }
            if seen[axis] {
                return Err(Error::DuplicateAxis { axis });
            }
            seen[axis] = true;
        }

        let shape = permuted_indices
            .iter()
            .map(|&i| self.shape.as_slice()[i])
            .collect();
        let strides = permuted_indices.iter().map(|&i| self.strides[i]).collect();

        Ok(Self {
            shape: TensorShape::new(shape),
            strides,
        })
    }

    /// Merge the dimensions in the specified inclusive range into a single
    /// dimension.
    ///
    /// # Arguments
    /// * `dim_range` - A range specifying the dimensions to merge.
    ///
    /// # Returns
    /// * A new `TensorLayout` with the merged dimensions.
    ///
    /// # Errors
    /// * Returns an error if the specified range is invalid.
    pub fn merge(&self, dim_range: RangeInclusive<usize>) -> Result<Self> {
        let (start, end) = (*dim_range.start(), *dim_range.end());
        let rank = self.shape.len();

        if !(start <= end && end < rank) {
            return Err(Error::InvalidMergeRange { start, end, rank });
        }

        // Compute the merged size by multiplying the sizes in the inclusive range.
        let merged_size = self.shape.as_slice()[start..=end].iter().product();
        let merged_stride = self.strides[end];

        let mut new_shape = Vec::with_capacity(self.shape.len() - (end - start));
        let mut new_strides = Vec::with_capacity(self.strides.len() - (end - start));

        new_shape.extend_from_slice(&self.shape.as_slice()[..start]);
        new_shape.push(merged_size);
        new_shape.extend_from_slice(&self.shape.as_slice()[end + 1..]);

        new_strides.extend_from_slice(&self.strides[..start]);
        new_strides.push(merged_stride);
        new_strides.extend_from_slice(&self.strides[end + 1..]);

        Ok(Self {
            shape:   TensorShape::new(new_shape),
            strides: new_strides,
        })
    }

    /// Split a given dimension in the tensor into multiple adjacent dimensions.
    ///
    /// # Arguments
    /// * `dim` - The dimension index to split.
    /// * `shape` - Any type implementing `AsRef<[usize]>` (for example
    ///   `&[usize]`, `Vec<usize>`, `&Vec<usize>`, or `&TensorShape`). Use `0`
    ///   as a wildcard to infer the size of that dimension.
    ///
    /// # Returns
    /// * A new `TensorLayout` with the specified dimension split into multiple
    ///   dimensions.
    ///
    /// # Errors
    /// * Returns an error if the specified dimension is out of bounds, if there
    ///   are multiple wildcards in the provided shape, or if the split sizes do
    ///   not multiply to the original dimension size.
    ///
    /// # Example
    /// ```rust
    /// let layout = TensorLayout::new(vec![12, 4]);
    /// // You can pass a slice directly
    /// let split_layout = layout.split(0, &[3, 4]).unwrap();
    /// assert_eq!(split_layout.shape().as_slice(), &[3, 4, 4]);
    ///
    /// // Or a `Vec<usize>`
    /// let split_layout2 = layout.split(0, vec![3, 4]).unwrap();
    /// assert_eq!(split_layout2.shape().as_slice(), &[3, 4, 4]);
    ///
    /// // Or a `&TensorShape` (because `TensorShape` implements `AsRef<[usize]>`)
    /// let ts = TensorShape::new(vec![3, 4]);
    /// let split_layout3 = layout.split(0, &ts).unwrap();
    /// assert_eq!(split_layout3.shape().as_slice(), &[3, 4, 4]);
    /// ```
    /// # Errors
    /// ```rust
    /// let layout = TensorLayout::new(vec![10, 4]);
    /// let err = layout.split(0, &[3, 4]).unwrap_err();
    /// match err {
    ///     Error::InvalidDimensionSplit { original_size, shape } => {
    ///         assert_eq!(original_size, 10);
    ///         assert_eq!(shape, &[3, 4]);
    ///     }
    ///     _ => panic!("unexpected error variant"),
    /// }
    /// ```
    pub fn split(&self, dim: usize, shape: impl AsRef<[usize]>) -> Result<Self> {
        if dim >= self.shape.len() {
            return Err(Error::AxisOutOfBounds { axis: dim, rank: self.shape.len() });
        }

        let original_size = self.shape[dim];
        let original_stride = self.strides[dim];

        // Calculate the product of non-zero sizes and identify wildcard index.
        let mut non_zero_product = 1usize;
        let mut zero_index = None;

        let shape = shape.as_ref();

        for (i, &size) in shape.iter().enumerate() {
            if size == 0 {
                if zero_index.is_some() {
                    return Err(Error::TooManySplitWildcards);
                }
                zero_index = Some(i);
            } else {
                non_zero_product *= size;
            }
        }

        // Determine the final sizes for the split dimensions, inferring wildcards.
        let mut final_sizes = shape.to_vec();
        if let Some(zero_index) = zero_index {
            if original_size % non_zero_product != 0 {
                return Err(Error::InvalidDimensionSplit { original_size, shape: final_sizes });
            }
            let inferred_size = original_size / non_zero_product;
            final_sizes[zero_index] = inferred_size;
        }

        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();

        // Add dimensions before the split dimension.
        new_shape.extend_from_slice(&self.shape.as_slice()[..dim]);
        new_strides.extend_from_slice(&self.strides[..dim]);

        // Compute strides for the new split dimensions.
        let mut current_stride = original_stride;
        for &size in final_sizes.iter().rev() {
            new_strides.push(current_stride);
            current_stride *= size;
        }

        // Reverse the strides for the split dimensions to maintain correct order.
        let start_idx = new_strides.len() - final_sizes.len();
        new_strides[start_idx..].reverse();

        // Add the sizes for the new split dimensions.
        new_shape.extend_from_slice(&final_sizes);

        // Add dimensions after the split dimension.
        if dim + 1 < self.shape().len() {
            new_shape.extend_from_slice(&self.shape().as_slice()[dim + 1..]);
            new_strides.extend_from_slice(&self.strides[dim + 1..]);
        }

        Ok(Self {
            shape:   TensorShape::new(new_shape),
            strides: new_strides,
        })
    }

    // pub fn reshape(&self, shape: &[usize]) -> Result<Self> {

    // }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorShape(Vec<usize>);

impl TensorShape {
    pub fn new(shape: Vec<usize>) -> Self {
        Self(shape)
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    pub fn size(&self) -> usize {
        self.0.iter().product()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'a> IntoIterator for &'a TensorShape {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Index<usize> for TensorShape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl AsRef<[usize]> for TensorShape {
    fn as_ref(&self) -> &[usize] {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_stride_correctly() {
        let shape = vec![2, 3, 4];
        let expected_stride = vec![12, 4, 1];
        let computed_stride = TensorLayout::compute_stride(&shape);
        assert_eq!(computed_stride, expected_stride);
    }

    // Tests for index un/ravelling

    #[test]
    fn ravel_index_shape_mismatch() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let indices = vec![1, 2]; // Incorrect length
        let err = layout.ravel_index(&indices).unwrap_err();
        match err {
            Error::RankMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn ravel_index_correctly() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let indices = vec![1, 2, 3];
        let expected_index = 1 * 12 + 2 * 4 + 3;
        let computed_index = layout.ravel_index(&indices).unwrap();
        assert_eq!(computed_index, expected_index);
    }

    #[test]
    fn unravel_index_empty_shape() {
        let layout = TensorLayout::new(vec![]);
        let index = 0;
        let expected_indices: Vec<usize> = vec![];
        let computed_indices = layout.unravel_index(index).unwrap();
        assert_eq!(computed_indices, expected_indices);
    }

    #[test]
    fn unravel_index_out_of_range() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let index = 24; // size is 24, so 24 is out of range
        let err = layout.unravel_index(index).unwrap_err();
        match err {
            Error::LinearIndexOutOfRange { index: idx, size } => {
                assert_eq!(idx, 24);
                assert_eq!(size, 24);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn unravel_index_correctly() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let index = 23;
        let expected_indices = vec![1, 2, 3];
        let computed_indices = layout.unravel_index(index).unwrap();
        assert_eq!(computed_indices, expected_indices);
    }

    // Tests for merge

    #[test]
    fn merge_invalid_range() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let res = layout.merge(2..=1).unwrap_err();
        match res {
            Error::InvalidMergeRange { start, end, rank } => {
                assert_eq!(start, 2);
                assert_eq!(end, 1);
                assert_eq!(rank, 3);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn merge_success() {
        let layout = TensorLayout::new(vec![2, 3, 4, 5]);
        // merge axes 1..=2 -> shape becomes [2, 12, 5]
        let merged = layout.merge(1..=2).unwrap();
        assert_eq!(merged.shape().as_slice(), &[2, 12, 5]);
    }

    // Tests for split

    #[test]
    fn split_axis_out_of_bounds() {
        let layout = TensorLayout::new(vec![2, 3]);
        let err = layout.split(2, &[1, 2]).unwrap_err();
        match err {
            Error::AxisOutOfBounds { axis, rank } => {
                assert_eq!(axis, 2);
                assert_eq!(rank, 2);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn split_too_many_wildcards() {
        let layout = TensorLayout::new(vec![12]);
        let err = layout.split(0, &[0, 0]).unwrap_err();
        match err {
            Error::TooManySplitWildcards => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn split_invalid_dimension_split_with_wildcard() {
        let layout = TensorLayout::new(vec![10]);
        let err = layout.split(0, &[3, 0]).unwrap_err();
        match err {
            Error::InvalidDimensionSplit { original_size, shape } => {
                assert_eq!(original_size, 10);
                assert_eq!(shape, &[3, 0]);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn split_success_exact() {
        let layout = TensorLayout::new(vec![12, 4]);
        let split = layout.split(0, &[3, 4]).unwrap();
        assert_eq!(split.shape().as_slice(), &[3, 4, 4]);

        // Verify ravel/unravel consistency with expected strides
        let expected_strides = TensorLayout::compute_stride(split.shape().as_slice());
        let idx = vec![2, 3, 1];
        let expected_flat: usize = idx
            .iter()
            .zip(expected_strides.iter())
            .fold(0usize, |acc, (i, s)| acc + i * s);
        assert_eq!(split.ravel_index(&idx).unwrap(), expected_flat);
        assert_eq!(split.unravel_index(expected_flat).unwrap(), idx);
    }

    #[test]
    fn split_success_with_wildcard() {
        let layout = TensorLayout::new(vec![12, 5]);
        let split = layout.split(0, &[0, 3]).unwrap();
        // original axis 0 (12) split into [4,3]
        assert_eq!(split.shape().as_slice(), &[4, 3, 5]);

        let expected_strides = TensorLayout::compute_stride(split.shape().as_slice());
        let idx = vec![3, 2, 4];
        let expected_flat: usize = idx
            .iter()
            .zip(expected_strides.iter())
            .fold(0usize, |acc, (i, s)| acc + i * s);
        assert_eq!(split.ravel_index(&idx).unwrap(), expected_flat);
        assert_eq!(split.unravel_index(expected_flat).unwrap(), idx);
    }

    #[test]
    fn split_middle_dim_preserves_flat_index() {
        // Original shape [2, 12, 5] -> split axis 1 into [3,4] -> [2,3,4,5]
        let orig = TensorLayout::new(vec![2, 12, 5]);
        let split = orig.split(1, &[3, 4]).unwrap();

        // pick an original index and its corresponding split indices
        let a = 1usize;
        let b = 7usize; // in [0..12)
        let c = 2usize;
        // decompose b into (x, y) where b = x * 4 + y
        let x = b / 4;
        let y = b % 4;

        let orig_idx = vec![a, b, c];
        let split_idx = vec![a, x, y, c];

        let orig_flat = orig.ravel_index(&orig_idx).unwrap();
        let split_flat = split.ravel_index(&split_idx).unwrap();
        assert_eq!(orig_flat, split_flat);

        // also check unravel round-trip for split layout
        let unr = split.unravel_index(split_flat).unwrap();
        assert_eq!(unr, split_idx);
    }
}
