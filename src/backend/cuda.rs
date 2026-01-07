#![allow(dead_code, unused)]

use super::Backend;
use crate::layout::TensorLayout;
use crate::storage::{CudaStorage, TensorStorage};
use crate::tensor::{DType, Tensor};
use std::{
    fmt::Debug,
    sync::{Arc, RwLock},
};

#[derive(Debug)]
struct CudaBackend;

impl Backend for CudaBackend {
    fn alloc_storage(&self, layout: &TensorLayout, dtype: DType) -> TensorStorage {
        unimplemented!()
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        unimplemented!()
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor {
        unimplemented!()
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        unimplemented!()
    }

    fn div(&self, a: &Tensor, b: &Tensor) -> Tensor {
        unimplemented!()
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        unimplemented!()
    }
}
