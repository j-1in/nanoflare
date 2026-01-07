use crate::tensor::DType;
use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
    sync::{Arc, RwLock},
};

#[derive(Debug, Clone)]
pub enum TensorStorage {
    Cpu(Arc<RwLock<CpuStorage>>),
    Cuda(Arc<RwLock<CudaStorage>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Index<usize> for CpuStorage {
    type Output =;

    fn index(&self, index: usize) -> &Self::Output {
        &self.
    }
}

impl IndexMut<usize> for CpuStorage {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        todo!()
    }
}

/// UNIMPLEMENTED
#[derive(Debug, Clone, PartialEq)]
pub enum CudaStorage {}
