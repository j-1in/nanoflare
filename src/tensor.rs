use crate::backend::Backend;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::TensorStorage;
use std::{
    fmt::Debug,
    ops::{Add, Div, Index, Mul, Sub},
    sync::Arc,
};

macro_rules! impl_binary_op {
    ($($trait:ident, $method:ident);* $(;)?) => {
        $(
            // 1. Owned implementation: a + b
            impl<T: DType> $trait for Tensor<T> {
                type Output = Self;
                fn $method(self, rhs: Self) -> Self::Output {
                    self.backend.$method(&self, &rhs)
                }
            }

            // 2. Reference implementation: &a + &b
            impl<'a, 'b, T: DType> $trait<&'b Tensor<T>> for &'a Tensor<T> {
                type Output = Tensor<T>;
                fn $method(self, rhs: &'b Tensor<T>) -> Self::Output {
                    self.backend.$method(self, rhs)
                }
            }
        )*
    };
}

#[derive(Debug, Clone)]
pub struct Tensor<T: DType> {
    storage:       TensorStorage<T>,
    layout:        TensorLayout,
    backend:       Arc<dyn Backend<T>>,
    requires_grad: bool,
    grad:          Option<Box<Tensor<T>>>,
}

impl_binary_op!(
    Add, add;
    Sub, sub;
    Mul, mul;
    Div, div
);

impl<T: DType> Tensor<T> {
    pub fn zeros(layout: TensorLayout, backend: Arc<dyn Backend<T>>) -> Self {
        let storage = backend.store_zeros(&layout);

        Tensor {
            storage,
            layout,
            backend,
            requires_grad: false,
            grad: None,
        }
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }
}

impl<T: DType> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.storage[self.layout.ravel_index(indices)]
    }
}
