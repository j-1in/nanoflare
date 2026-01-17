use std::fmt::Debug;
use std::ops::{Add, Div, Index, Mul, RangeInclusive, Sub};
use std::sync::Arc;

use crate::Result;
use crate::backend::Backend;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::TensorStorage;

macro_rules! impl_binary_op {
    ($($trait:ident, $method:ident);* $(;)?) => {
        $(
            // 1. Owned implementation: a + b
            impl<T:DType, B: Backend<T>> $trait for Tensor<T, B> {
                type Output = Self;
                fn $method(self, rhs: Self) -> Self::Output {
                    self.backend.$method(&self, &rhs)
                }
            }

            // 2. Reference implementation: &a + &b
            impl<'a, 'b, T:DType, B: Backend<T>> $trait<&'b Tensor<T, B>> for &'a Tensor<T, B> {
                type Output = Tensor<T, B>;
                fn $method(self, rhs: &'b Tensor<T, B>) -> Self::Output {
                    self.backend.$method(self, rhs)
                }
            }
        )*
    };
}

#[derive(Debug, Clone)]
pub struct Tensor<T: DType, B: Backend<T>> {
    storage:       TensorStorage<T>,
    layout:        TensorLayout,
    backend:       Arc<B>,
    requires_grad: bool,
}

impl_binary_op!(
    Add, add;
    Sub, sub;
    Mul, mul;
    Div, div
);

impl<T: DType, B: Backend<T>> Tensor<T, B> {
    /// Create a new tensor filled with zeros, given a specific layout and
    /// backend.
    pub fn zeros(layout: TensorLayout, backend: Arc<B>) -> Self {
        let storage = backend.store_zeros(&layout);

        Tensor {
            storage,
            layout,
            backend,
            requires_grad: false,
        }
    }

    /// Get a reference to the tensor's layout.
    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    // TODO: avoid unnecessary clone
    /// Permute the dimensions of the tensor layout according to the given
    /// indices and return a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::permute`.
    /// `Tensor`
    pub fn permute(&self, permuted_indices: &[usize]) -> Result<Self> {
        let layout = self.layout.permute(permuted_indices)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad.clone(),
        })
    }

    /// Merge the dimensions in the specified inclusive range into a single
    /// dimension and return a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::merge`.
    pub fn merge(&self, dim_range: RangeInclusive<usize>) -> Result<Self> {
        let layout = self.layout.merge(dim_range)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad.clone(),
        })
    }

    /// Split a given dimension in the tensor into multiple adjacent dimensions
    /// with specified sizes, returning a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::split`.
    pub fn split(&self, dim: usize, shape: impl AsRef<[usize]>) -> Result<Self> {
        let layout = self.layout.split(dim, shape)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad.clone(),
        })
    }

    /// Reshape the tensor layout to a new shape if it is contiguous in memory
    /// according to `TensorLayout::is_contiguous`. Otherwise, this operation
    /// should create a new contiguous copy and reshape that copy, TODO.
    pub fn reshape(&self, shape: impl AsRef<[usize]>) -> Result<Self> {
        if self.layout.is_contiguous() {
            let layout = self.layout.reshape(shape)?;

            Ok(Tensor {
                layout,
                storage: self.storage.clone(),
                backend: self.backend.clone(),
                requires_grad: self.requires_grad.clone(),
            })
        } else {
            todo!("non-contiguous reshape not implemented yet")
        }
    }

    /// Slice the tensor along a single dimension, producing a sub-layout and
    /// returning a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::slice`.
    pub fn slice(&self, dim: usize, range: RangeInclusive<usize>) -> Result<Self> {
        let layout = self.layout.slice(dim, range)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad.clone(),
        })
    }

    /// Create a strided view by skipping elements along a single axis with
    /// the given step size, returning a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::skip`.
    pub fn skip(&self, dim: usize, step: usize) -> Result<Self> {
        let layout = self.layout.skip(dim, step)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad.clone(),
        })
    }
}

impl<T: DType, B: Backend<T>> Index<&[usize]> for Tensor<T, B> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        let idx = self
            .layout
            .ravel_index(indices)
            .expect("invalid index for tensor");
        &self.storage[idx]
    }
}
