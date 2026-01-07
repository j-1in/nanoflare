use super::Backend;
use crate::layout::TensorLayout;
use crate::storage::{CpuStorage, TensorStorage};
use crate::tensor::{DType, Tensor};
use std::{
    fmt::Debug,
    sync::{Arc, RwLock},
};

#[derive(Debug)]
struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn alloc_storage(&self, layout: &TensorLayout, dtype: DType) -> TensorStorage {
        let size = layout.shape().size();

        let storage = match dtype {
            DType::U8 => CpuStorage::U8(vec![0u8; size]),
            DType::U16 => CpuStorage::U16(vec![0u16; size]),
            DType::U32 => CpuStorage::U32(vec![0u32; size]),
            DType::U64 => CpuStorage::U64(vec![0u64; size]),
            DType::I8 => CpuStorage::I8(vec![0i8; size]),
            DType::I16 => CpuStorage::I16(vec![0i16; size]),
            DType::I32 => CpuStorage::I32(vec![0i32; size]),
            DType::I64 => CpuStorage::I64(vec![0i64; size]),
            DType::F32 => CpuStorage::F32(vec![0f32; size]),
            DType::F64 => CpuStorage::F64(vec![0f64; size]),
        };

        TensorStorage::Cpu(Arc::new(RwLock::new(storage)))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn div(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_addition() {
        let backend = Arc::new(crate::backend::cpu::CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 2]);
        let a = Tensor::zeros(layout.clone(), DType::F32, backend.clone());
        let b = Tensor::zeros(layout.clone(), DType::F32, backend.clone());
        let c = &a + &b;

        assert_eq!(c.layout(), &layout);
    }
}
