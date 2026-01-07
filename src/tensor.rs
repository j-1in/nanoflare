use crate::backend::Backend;
use crate::layout::TensorLayout;
use crate::storage::TensorStorage;
use std::{
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    sync::Arc,
};

macro_rules! impl_binary_op {
    ($($trait:ident, $method:ident);* $(;)?) => {
        $(
            // 1. Owned implementation: a + b
            impl $trait for Tensor {
                type Output = Self;
                fn $method(self, rhs: Self) -> Self::Output {
                    self.backend.$method(&self, &rhs)
                }
            }

            // 2. Reference implementation: &a + &b
            impl<'a, 'b> $trait<&'b Tensor> for &'a Tensor {
                type Output = Tensor;
                fn $method(self, rhs: &'b Tensor) -> Self::Output {
                    self.backend.$method(self, rhs)
                }
            }
        )*
    };
}

#[rustfmt::skip]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DType {
    U8, U16, U32, U64, I8, I16, I32, I64, F32, F64,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    storage:       TensorStorage,
    layout:        TensorLayout,
    dtype:         DType,
    backend:       Arc<dyn Backend>,
    requires_grad: bool,
    grad:          Option<Box<Tensor>>,
}

impl_binary_op!(
    Add, add;
    Sub, sub;
    Mul, mul;
    Div, div
);

impl Tensor {
    pub fn zeros(layout: TensorLayout, dtype: DType, backend: Arc<dyn Backend>) -> Self {
        let storage = backend.alloc_storage(&layout, dtype);

        Tensor {
            storage,
            layout,
            dtype,
            backend,
            requires_grad: false,
            grad: None,
        }
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }
}
