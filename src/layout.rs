use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

#[derive(Debug, Clone, PartialEq)]
pub struct TensorLayout {
    shape:  TensorShape,
    stride: Vec<usize>,
}

impl TensorLayout {
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    pub fn new(shape: Vec<usize>) -> Self {
        let stride = Self::compute_stride(&shape);

        TensorLayout {
            shape: TensorShape::new(shape),
            stride,
        }
    }

    pub fn compute_stride(shape: &[usize]) -> Vec<usize> {
        let mut stride = vec![0; shape.len()];
        let mut prod = 1;

        for (i, dim) in shape.iter().rev().enumerate() {
            stride[shape.len() - 1 - i] = prod;
            prod *= *dim;
        }

        stride
    }

    pub fn ravel_index(&self, indices: &[usize]) -> usize {
        if indices.len() != self.shape.len() {
            panic!("Indices length does not match tensorshape dimensions.");
        }

        indices
            .iter()
            .zip(&self.stride)
            .fold(0usize, |acc, (idx, stride)| acc + idx * stride)
    }

    /// FIXME
    pub fn unravel_index(&self, index: usize) -> Vec<usize> {
        if self.shape.is_empty() {
            return vec![];
        }

        let mut indices = vec![0; self.shape.len()];
        let mut remainder = index;

        for i in 0..self.shape.len() {
            indices[i] = remainder / self.stride[i];
            remainder %= self.stride[i];
        }

        indices
    }
}

// #[derive(Debug, Clone, PartialEq)]
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
