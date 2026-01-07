#![allow(dead_code, unused)]

use super::Backend;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::{CudaStorage, TensorStorage};
use crate::tensor::Tensor;
use std::{
    fmt::Debug,
    sync::{Arc, RwLock},
};

#[derive(Debug)]
struct CudaBackend;

impl<T: DType> Backend<T> for CudaBackend {
    fn alloc_storage(&self, layout: &TensorLayout) -> TensorStorage<T> {
        unimplemented!()
    }

    fn add(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        unimplemented!()
    }

    fn sub(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        unimplemented!()
    }

    fn mul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        unimplemented!()
    }

    fn div(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        unimplemented!()
    }

    fn matmul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        unimplemented!()
    }
}
