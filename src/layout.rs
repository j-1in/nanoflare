use crate::Result;
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

    pub fn ravel_index(&self, indices: &[usize]) -> usize {
        if indices.len() != self.shape.len() {
            panic!("Indices length does not match tensorshape dimensions.");
        }

        indices
            .iter()
            .zip(&self.strides)
            .fold(0usize, |acc, (idx, stride)| acc + idx * stride)
    }

    pub fn unravel_index(&self, index: usize) -> Vec<usize> {
        if self.shape.is_empty() {
            return vec![];
        }

        let mut indices = vec![0; self.shape.len()];
        let mut remainder = index;

        for i in 0..self.shape.len() {
            indices[i] = remainder / self.strides[i];
            remainder %= self.strides[i];
        }

        indices
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
            return Err(crate::Error::RankMismatch {
                expected: rank,
                got:      permuted_indices.len(),
            });
        }

        // 2. Check bounds + duplicates
        let mut seen = vec![false; rank];
        for &axis in permuted_indices {
            if axis >= rank {
                return Err(crate::Error::AxisOutOfBounds { axis, rank });
            }
            if seen[axis] {
                return Err(crate::Error::DuplicateAxis { axis });
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

    pub fn merge(&self, dim_range: RangeInclusive<usize>) -> Self {
        let (start, end) = (*dim_range.start(), *dim_range.end());

        assert!(
            start <= end && end < self.shape.len(),
            "Invalid dimension range for merge."
        );

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

        Self {
            shape:   TensorShape::new(new_shape),
            strides: new_strides,
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

// impl Index<usize> for TensorShape {
//     type Output = usize;
//     fn index(&self, index: usize) -> &Self::Output {
//         &self.0[index]
//     }
// }

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

    #[test]
    #[should_panic(expected = "Indices length does not match tensorshape dimensions.")]
    fn ravel_index_shape_mismatch() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let indices = vec![1, 2]; // Incorrect length
        layout.ravel_index(&indices);
    }

    #[test]
    fn ravel_index_correctly() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let indices = vec![1, 2, 3];
        let expected_index = 1 * 12 + 2 * 4 + 3;
        let computed_index = layout.ravel_index(&indices);
        assert_eq!(computed_index, expected_index);
    }

    #[test]
    fn unravel_index_empty_shape() {
        let layout = TensorLayout::new(vec![]);
        let index = 0;
        let expected_indices: Vec<usize> = vec![];
        let computed_indices = layout.unravel_index(index);
        assert_eq!(computed_indices, expected_indices);
    }

    #[test]
    fn unravel_index_correctly() {
        let layout = TensorLayout::new(vec![2, 3, 4]);
        let index = 23;
        let expected_indices = vec![1, 2, 3];
        let computed_indices = layout.unravel_index(index);
        assert_eq!(computed_indices, expected_indices);
    }
}
