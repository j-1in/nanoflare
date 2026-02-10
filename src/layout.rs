use std::fmt::Debug;
use std::ops::{Index, RangeInclusive};

use crate::index::TensorIndex;
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct TensorLayout {
    shape:   TensorShape,
    strides: Vec<usize>,
    offset:  usize,
}

impl TensorLayout {
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn new(shape: Vec<usize>) -> Self {
        let strides = Self::compute_stride(&shape);

        TensorLayout {
            shape: TensorShape::new(shape),
            strides,
            offset: 0,
        }
    }

    // Main entry point to index a tensor with a set of `TensorIndex` values.
    pub fn get<I: AsRef<[TensorIndex]>>(&self, indices: I) -> Result<Self> {
        let expanded = self.expand_indices(indices.as_ref())?;
        self.view_from_expanded(&expanded)
    }

    // Expand the indices, handling the `Rest` variant.
    fn expand_indices(&self, indices: &[TensorIndex]) -> Result<Vec<TensorIndex>> {
        let rank = self.shape.len();

        let mut expanded = Vec::new();
        let mut rest_used = false;

        for idx in indices {
            match idx {
                TensorIndex::Rest => {
                    if rest_used {
                        return Err(Error::RankMismatch {
                            expected: rank,
                            got:      indices.len(),
                        });
                    }
                    rest_used = true;

                    let specified = indices.len().saturating_sub(1);
                    if specified > rank {
                        return Err(Error::RankMismatch {
                            expected: rank,
                            got:      indices.len(),
                        });
                    }

                    let rest_len = rank - specified;
                    expanded.extend(
                        std::iter::repeat(TensorIndex::Slice { start: None, end: None })
                            .take(rest_len),
                    );
                }
                _ => expanded.push(*idx),
            }
        }

        if !rest_used && indices.len() != rank {
            return Err(Error::RankMismatch {
                expected: rank,
                got:      indices.len(),
            });
        }

        if expanded.len() != rank {
            return Err(Error::RankMismatch {
                expected: rank,
                got:      expanded.len(),
            });
        }

        Ok(expanded)
    }

    // Create a new view of the tensor based on the expanded indices.
    fn view_from_expanded(&self, expanded: &[TensorIndex]) -> Result<Self> {
        let rank = self.shape.len();
        if expanded.len() != rank {
            return Err(Error::RankMismatch {
                expected: rank,
                got:      expanded.len(),
            });
        }

        let mut new_shape = Vec::with_capacity(rank);
        let mut new_strides = Vec::with_capacity(rank);
        let mut new_offset = self.offset;

        for (axis, idx) in expanded.iter().enumerate() {
            let dim = self.shape[axis];
            let stride = self.strides[axis];

            match idx {
                TensorIndex::Single(index) => {
                    if *index >= dim {
                        return Err(Error::IndexOutOfBounds { axis, index: *index, dim });
                    }

                    new_offset += index * stride;
                    new_shape.push(1);
                    new_strides.push(stride);
                }
                TensorIndex::Slice { start, end } => {
                    let s = start.unwrap_or(0);
                    let e = end.unwrap_or(dim);

                    if s > e || s > dim || e > dim {
                        return Err(Error::IndexOutOfBounds {
                            axis,
                            index: if e > dim { e } else { s },
                            dim,
                        });
                    }

                    new_offset += s * stride;
                    new_shape.push(e - s);
                    new_strides.push(stride);
                }
                TensorIndex::Rest => {
                    return Err(Error::RankMismatch {
                        expected: rank,
                        got:      expanded.len(),
                    });
                }
            }
        }

        Ok(Self {
            shape:   TensorShape::new(new_shape),
            strides: new_strides,
            offset:  new_offset,
        })
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

        // Include any layout offset so the returned linear index refers to the
        // underlying buffer position.
        Ok(self.offset + flat)
    }

    /// Convert a flat (raveled) index into multi-dimensional indices.
    ///
    /// Returns an error if the flat index is out of range for the tensor size.
    pub fn unravel_index(&self, index: usize) -> Result<Vec<usize>> {
        if self.shape.is_empty() {
            return Ok(vec![]);
        }

        let total = self.shape.numel();
        // The provided `index` is expected to be an index into the underlying
        // buffer; it must fall within the window described by `offset .. offset +
        // total`.
        if index < self.offset || index >= self.offset + total {
            return Err(Error::LinearIndexOutOfRange { index, size: total });
        }

        let mut indices = vec![0; self.shape.len()];
        let mut remainder = index - self.offset;

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
    /// A new `TensorLayout` with permuted shape and strides.
    ///
    /// # Example
    /// ```rust
    /// use nanoflare::layout::TensorLayout;
    /// let layout = TensorLayout::new(vec![2, 3, 4]);
    /// let permuted_layout = layout.permute(&[2, 0, 1]).unwrap();
    /// assert_eq!(permuted_layout.shape().as_slice(), &[4, 2, 3]);
    /// ```
    ///
    /// # Errors
    /// Returns an error if `permuted_indices` is not a valid permutation of
    /// the tensor dimensions.
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
            offset: self.offset,
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
            offset:  self.offset,
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
    /// use nanoflare::layout::{TensorLayout, TensorShape};
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
    /// use nanoflare::Error;
    /// use nanoflare::layout::TensorLayout;
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
        } else {
            // No wildcard provided: the product of the provided sizes must exactly
            // equal the original dimension size.
            if non_zero_product != original_size {
                return Err(Error::InvalidDimensionSplit { original_size, shape: final_sizes });
            }
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
            offset:  self.offset,
        })
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for (&dim, &stride) in self
            .shape
            .as_slice()
            .iter()
            .rev()
            .zip(self.strides.iter().rev())
        {
            if dim == 0 {
                continue; // Skip zero-sized dimensions
            }
            if stride != expected_stride {
                return false;
            }
            expected_stride *= dim;
        }
        true
    }

    /// Reshape the tensor layout to a new shape.
    ///
    /// The total number of elements must remain the same. This operation only
    /// changes the logical shape and recomputes canonical (C-contiguous)
    /// strides for the new shape; it does not modify the underlying memory or
    /// attempt to preserve non-contiguous memory layouts. The current `offset`
    /// is preserved.
    ///
    /// # Arguments
    /// * `shape` - Any type implementing `AsRef<[usize]>` describing the new
    ///   shape. The product of the new dimensions must equal the current tensor
    ///   size.
    ///
    /// # Returns
    /// A new `TensorLayout` with the requested shape and newly computed
    /// contiguous strides.
    ///
    /// # Errors
    /// Returns `Error::TensorSizeMismatch` if the requested shape contains a
    /// different number of elements than the current layout.
    pub fn reshape(&self, shape: impl AsRef<[usize]>) -> Result<Self> {
        let shape = shape.as_ref();

        if shape.iter().product::<usize>() != self.shape().numel() {
            return Err(Error::TensorSizeMismatch {
                expected: self.shape().as_slice().to_vec(),
                got:      shape.to_vec(),
            });
        }

        Ok(Self {
            shape:   TensorShape::new(shape.to_vec()),
            strides: Self::compute_stride(shape),
            offset:  self.offset,
        })
    }

    /// Broadcast the tensor layout to a new shape. Where broadcasted dimensions
    /// will have stride 0 in the resulting layout.
    ///
    /// # Arguments
    /// * `shape` - Any type implementing `AsRef<[usize]>` describing the target
    ///   shape. The target shape must be broadcastable from the current shape.
    ///
    /// # Returns
    /// A new `TensorLayout` with the requested shape and strides adjusted for
    /// broadcasting.
    ///
    /// # Errors
    /// Returns `Error::LayoutMismatch` if the target shape is not broadcastable
    /// from the current shape.
    pub fn broadcast_to(&self, shape: impl AsRef<[usize]>) -> Result<Self> {
        let out_shape = shape.as_ref();
        let out_rank = out_shape.len();
        let in_shape = self.shape.as_slice();
        let in_rank = in_shape.len();

        if out_rank < in_rank {
            return Err(Error::LayoutMismatch {
                a: self.shape.clone(),
                b: TensorShape::new(out_shape.to_vec()),
            });
        }

        let lead = out_rank - in_rank;
        let mut new_shape = Vec::with_capacity(out_rank);
        let mut new_strides = Vec::with_capacity(out_rank);

        for i in 0..out_rank {
            let out_dim = out_shape[i];
            let in_dim_opt = if i >= lead {
                Some(in_shape[i - lead])
            } else {
                None
            };
            let in_dim = in_dim_opt.unwrap_or(1);

            if in_dim != out_dim && in_dim != 1 {
                return Err(Error::LayoutMismatch {
                    a: self.shape.clone(),
                    b: TensorShape::new(out_shape.to_vec()),
                });
            }

            let stride = if let Some(in_dim) = in_dim_opt {
                if in_dim == 1 && out_dim != 1 {
                    0
                } else {
                    self.strides[i - lead]
                }
            } else {
                0
            };

            new_shape.push(out_dim);
            new_strides.push(stride);
        }

        Ok(Self {
            shape:   TensorShape::new(new_shape),
            strides: new_strides,
            offset:  self.offset,
        })
    }

    /// Slice the tensor along a single dimension, producing a sub-layout.
    ///
    /// This operation does not copy memory; it adjusts the shape and offset so
    /// that the returned layout describes a view into the original tensor.
    /// Strides are preserved so indexing into the returned layout maps to the
    /// correct positions in the original buffer.
    ///
    /// # Arguments
    /// * `dim` - The axis to slice.
    /// * `range` - An inclusive range [start..=end] specifying the slice along
    ///   the chosen axis. Both `start` and `end` must be within the bounds of
    ///   that axis and `start` must be <= `end`.
    ///
    /// # Returns
    /// A new `TensorLayout` describing the sliced view. The `offset` is moved
    /// forward by `start * stride[dim]` and the size of the sliced axis is set
    /// to `end - start + 1`.
    ///
    /// # Errors
    /// Returns `Error::AxisOutOfBounds` if `dim` is not a valid axis, or
    /// `Error::IndexOutOfBounds` if the provided range is invalid for the
    /// selected axis.
    pub fn slice(&self, dim: usize, range: RangeInclusive<usize>) -> Result<Self> {
        if dim >= self.shape.len() {
            return Err(Error::AxisOutOfBounds { axis: dim, rank: self.shape.len() });
        }

        let (start, end) = (*range.start(), *range.end());
        let dim_size = self.shape[dim];

        if !(start <= end && end < dim_size) {
            return Err(Error::IndexOutOfBounds {
                axis:  dim,
                index: end,
                dim:   dim_size,
            });
        }

        let mut new_shape = self.shape.as_slice().to_vec();
        new_shape[dim] = end - start + 1;

        let additional_offset = start * self.strides[dim];

        Ok(Self {
            shape:   TensorShape::new(new_shape),
            strides: self.strides.clone(),
            offset:  self.offset + additional_offset,
        })
    }

    /// Create a strided view by skipping elements along a single axis.
    ///
    /// This returns a new layout where `stride[dim]` is multiplied by `step`,
    /// effectively taking every `step`-th element along the given axis. The
    /// length of that axis is rounded up using integer ceiling so the view
    /// covers all remaining elements when the axis length is not a multiple of
    /// `step`.
    ///
    /// This operation does not reallocate or copy data; it only modifies the
    /// logical strides and shape and preserves the existing `offset`.
    ///
    /// # Arguments
    /// * `dim` - The axis to apply skipping on.
    /// * `step` - The stride multiplier (must be >= 1). A `step` of 1 returns
    ///   an identical layout.
    ///
    /// # Returns
    /// A new `TensorLayout` representing the strided view.
    ///
    /// # Errors
    /// Returns `Error::AxisOutOfBounds` if `dim` is invalid or
    /// `Error::InvalidSkipStep` if `step` is zero.
    pub fn skip(&self, dim: usize, step: usize) -> Result<Self> {
        if dim >= self.shape.len() {
            return Err(Error::AxisOutOfBounds { axis: dim, rank: self.shape.len() });
        }

        if step == 0 {
            return Err(Error::InvalidSkipStep { step });
        }

        let mut new_strides = self.strides.clone();
        new_strides[dim] *= step;

        let mut new_shape = self.shape.clone();
        new_shape.0[dim] = new_shape.0[dim].div_ceil(step);

        Ok(Self {
            shape:   new_shape,
            strides: new_strides,
            offset:  self.offset,
        })
    }

    pub fn contiguous(&self) -> Self {
        Self {
            shape:   self.shape.clone(),
            strides: Self::compute_stride(self.shape.as_slice()),
            offset:  self.offset,
        }
    }
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

    pub fn numel(&self) -> usize {
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

    // New tests for reshape, slice, and skip

    #[test]
    fn reshape_preserves_size_and_recomputes_strides() {
        let layout = TensorLayout::new(vec![2, 3, 4]); // size 24
        let reshaped = layout.reshape(&[4, 6]).unwrap();
        assert_eq!(reshaped.shape().as_slice(), &[4, 6]);
        // reshaped should be canonical (contiguous)
        assert!(reshaped.is_contiguous());
    }

    #[test]
    fn reshape_ravel_unravel_consistency() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let reshaped = layout.reshape(&[4, 6]).unwrap();

        // Verify ravel/unravel consistency using canonical strides
        let expected_strides = TensorLayout::compute_stride(reshaped.shape().as_slice());
        let idx = vec![3, 5];
        let expected_flat: usize = idx
            .iter()
            .zip(expected_strides.iter())
            .fold(0usize, |acc, (i, s)| acc + i * s);
        assert_eq!(reshaped.ravel_index(&idx).unwrap(), expected_flat);
        assert_eq!(reshaped.unravel_index(expected_flat).unwrap(), idx);
    }

    #[test]
    fn reshape_size_mismatch_error() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let err = layout.reshape(&[5, 5]).unwrap_err();
        match err {
            Error::TensorSizeMismatch { expected, got } => {
                assert_eq!(expected, vec![2, 3, 4]);
                assert_eq!(got, vec![5, 5]);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn reshape_from_noncontiguous_layout_returns_contiguous_strides() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let permuted = layout.permute(&[2, 1, 0]).unwrap();
        assert!(!permuted.is_contiguous());
        // reshape will recompute canonical strides, ignoring non-contiguity
        let reshaped = permuted.reshape(&[4, 6]).unwrap();
        assert!(reshaped.is_contiguous());
    }

    #[test]
    fn slice_axis_out_of_bounds() {
        let layout = TensorLayout::new(vec![4, 5, 6]);
        let err = layout.slice(3, 0..=1).unwrap_err();
        match err {
            Error::AxisOutOfBounds { axis, rank } => {
                assert_eq!(axis, 3);
                assert_eq!(rank, 3);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn slice_invalid_range() {
        let layout = TensorLayout::new(vec![4, 5, 6]);
        let err = layout.slice(1, 2..=5).unwrap_err();
        match err {
            Error::IndexOutOfBounds { axis, index, dim } => {
                assert_eq!(axis, 1);
                assert_eq!(index, 5);
                assert_eq!(dim, 5);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn slice_valid_shape_and_offset_preserved() {
        let layout = TensorLayout::new(vec![4, 5, 6]);
        let s = layout.slice(1, 1..=3).unwrap();
        assert_eq!(s.shape().as_slice(), &[4, 3, 6]);
        // strides preserved
        assert_eq!(s.strides, layout.strides);
        // offset advanced by start * stride[1]
        assert_eq!(s.offset, layout.offset + 1 * layout.strides[1]);
    }

    #[test]
    fn slice_ravel_unravel_consistency() {
        let layout = TensorLayout::new(vec![4, 5, 6]);
        let s = layout.slice(1, 1..=3).unwrap();
        let idx = vec![2, 1, 5]; // within [4,3,6]
        let sliced_flat = s.ravel_index(&idx).unwrap();
        // corresponding original index: second axis needs offset by start
        let orig_idx = vec![2, 1 + 1, 5];
        let orig_flat = layout.ravel_index(&orig_idx).unwrap();
        assert_eq!(sliced_flat, orig_flat);
    }

    #[test]
    fn skip_axis_out_of_bounds_error() {
        let layout = TensorLayout::new(vec![3, 6]);
        let err = layout.skip(2, 2).unwrap_err();
        match err {
            Error::AxisOutOfBounds { axis, rank } => {
                assert_eq!(axis, 2);
                assert_eq!(rank, 2);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn skip_invalid_step_error() {
        let layout = TensorLayout::new(vec![3, 6]);
        let err = layout.skip(1, 0).unwrap_err();
        match err {
            Error::InvalidSkipStep { step } => {
                assert_eq!(step, 0);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn skip_step_one_returns_identical() {
        let layout = TensorLayout::new(vec![3, 6]);
        let same = layout.skip(0, 1).unwrap();
        assert_eq!(same, layout);
    }

    #[test]
    fn skip_shape_and_stride_changes() {
        let layout = TensorLayout::new(vec![3, 6]);
        let skipped = layout.skip(1, 2).unwrap();
        // axis length: ceil(6/2)=3
        assert_eq!(skipped.shape().as_slice(), &[3, 3]);
        // strides: original stride[1] * 2
        assert_eq!(skipped.strides[1], layout.strides[1] * 2);
    }

    #[test]
    fn skip_ravel_mapping_consistency() {
        let layout = TensorLayout::new(vec![3, 6]);
        let skipped = layout.skip(1, 2).unwrap();
        // ravel mapping: an index (a,b) in skipped layout corresponds to (a, b*2) in
        // original
        let a = 2usize;
        let b = 2usize; // in skipped axis range [0..3)
        let skipped_idx = vec![a, b];
        let skipped_flat = skipped.ravel_index(&skipped_idx).unwrap();
        let orig_idx = vec![a, b * 2];
        let orig_flat = layout.ravel_index(&orig_idx).unwrap();
        assert_eq!(skipped_flat, orig_flat);
    }

    #[test]
    fn skip_rounding_behavior_last_element_maps() {
        let l = TensorLayout::new(vec![5]);
        let sk = l.skip(0, 2).unwrap();
        assert_eq!(sk.shape().as_slice(), &[3]);
        // last element maps to original index 4
        let last_flat = sk.ravel_index(&[2usize]).unwrap();
        let orig_last_flat = l.ravel_index(&[4usize]).unwrap();
        assert_eq!(last_flat, orig_last_flat);
    }

    // Tests for broadcast_to

    #[test]
    fn broadcast_to_same_shape_preserves_strides() {
        let layout = TensorLayout::new(vec![2, 3]);
        let out = layout.broadcast_to([2, 3]).unwrap();
        assert_eq!(out.shape().as_slice(), &[2, 3]);
        assert_eq!(out.strides(), layout.strides());
        assert_eq!(out.offset(), layout.offset());
    }

    #[test]
    fn broadcast_to_expands_with_zero_stride() {
        let layout = TensorLayout::new(vec![1, 3]);
        let out = layout.broadcast_to([2, 3]).unwrap();
        assert_eq!(out.shape().as_slice(), &[2, 3]);
        assert_eq!(out.strides(), &[0, 1]);
    }

    #[test]
    fn broadcast_to_leading_dimension() {
        let layout = TensorLayout::new(vec![3]);
        let out = layout.broadcast_to([2, 3]).unwrap();
        assert_eq!(out.shape().as_slice(), &[2, 3]);
        assert_eq!(out.strides(), &[0, 1]);
    }

    #[test]
    fn broadcast_to_incompatible_shape_errors() {
        let layout = TensorLayout::new(vec![2, 3]);
        let err = layout.broadcast_to([3, 2]).unwrap_err();
        match err {
            Error::LayoutMismatch { .. } => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn broadcast_to_smaller_rank_errors() {
        let layout = TensorLayout::new(vec![1, 3]);
        let err = layout.broadcast_to([3]).unwrap_err();
        match err {
            Error::LayoutMismatch { .. } => {}
            _ => panic!("unexpected error variant"),
        }
    }
}
