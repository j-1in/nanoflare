use std::fmt::Debug;
use std::sync::Arc;

use crate::backend::Backend;
use crate::dtype::DType;
use crate::{Result, Tensor, TensorShape};

pub trait TensorOp<T: DType, B: Backend<T>>: Debug + Send + Sync {
    /// Returns the name of the operation.
    fn name(&self) -> &str;

    // FIXME
    fn backward(
        &self,
        inputs: &[&Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>>;

    fn to_optype(&self) -> OpType;
}

pub trait BinaryOp<T: DType, B: Backend<T>>: TensorOp<T, B> {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self>
    where
        Self: Sized;
    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct AddOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for AddOp {
    fn name(&self) -> &str {
        "Add"
    }

    fn backward(
        &self,
        inputs: &[&Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        // let lhs_grad = grad.sum_to_shape(&self.lhs_orig_shape, backend)?;
        // let rhs_grad = grad.sum_to_shape(&self.rhs_orig_shape, backend)?;
        // Ok(vec![lhs_grad, rhs_grad])
        todo!()
    }

    fn to_optype(&self) -> OpType {
        OpType::Add(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for AddOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(AddOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        shapes_match(a, b)?; // TODO: implement broadcasting rules
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for SubOp {
    fn name(&self) -> &str {
        "Sub"
    }

    fn backward(
        &self,
        inputs: &[&Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        // let lhs_grad = grad.sum_to_shape(&self.lhs_orig_shape, backend)?;
        // let rhs_grad = grad.sum_to_shape(&self.rhs_orig_shape, backend)?;
        // Ok(vec![lhs_grad, rhs_grad])
        todo!()
    }

    fn to_optype(&self) -> OpType {
        OpType::Sub(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for SubOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(SubOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        shapes_match(a, b)?; // TODO: implement broadcasting rules
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MulOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for MulOp {
    fn name(&self) -> &str {
        "Mul"
    }

    fn backward(
        &self,
        inputs: &[&Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        // let lhs_grad = grad.sum_to_shape(&self.lhs_orig_shape, backend)?;
        // let rhs_grad = grad.sum_to_shape(&self.rhs_orig_shape, backend)?;
        // Ok(vec![lhs_grad, rhs_grad])
        todo!()
    }

    fn to_optype(&self) -> OpType {
        OpType::Mul(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for MulOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(MulOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        shapes_match(a, b)?; // TODO: implement broadcasting rules
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DivOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for DivOp {
    fn name(&self) -> &str {
        "Div"
    }

    fn backward(
        &self,
        inputs: &[&Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        // let lhs_grad = grad.sum_to_shape(&self.lhs_orig_shape, backend)?;
        // let rhs_grad = grad.sum_to_shape(&self.rhs_orig_shape, backend)?;
        // Ok(vec![lhs_grad, rhs_grad])
        todo!()
    }

    fn to_optype(&self) -> OpType {
        OpType::Div(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for DivOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(DivOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        shapes_match(a, b)?; // TODO: implement broadcasting rules
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatMulOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for MatMulOp {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn backward(
        &self,
        inputs: &[&Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        // let lhs_grad = grad.sum_to_shape(&self.lhs_orig_shape, backend)?;
        // let rhs_grad = grad.sum_to_shape(&self.rhs_orig_shape, backend)?;
        // Ok(vec![lhs_grad, rhs_grad])
        todo!()
    }

    fn to_optype(&self) -> OpType {
        OpType::MatMul(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for MatMulOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(MatMulOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        // Validate that the inner dimensions match for matrix multiplication
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    Leaf,
    Add(AddOp),
    Sub(SubOp),
    Mul(MulOp),
    Div(DivOp),
    MatMul(MatMulOp),
}

pub(crate) fn shapes_match<T, B>(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()>
where
    T: crate::dtype::DType,
    B: crate::backend::Backend<T>,
{
    if a.layout().shape() != b.layout().shape() {
        return Err(crate::Error::LayoutMismatch {
            a: a.layout().shape().clone(),
            b: b.layout().shape().clone(),
        });
    }
    Ok(())
}

impl OpType {
    // pub fn validate_binop<T, B>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) ->
    // Result<()> where
    //     T: crate::dtype::DType,
    //     B: crate::backend::Backend<T>,
    // {
    //     match self {
    //         OpType::Add | OpType::Sub | OpType::Mul | OpType::Div =>
    // shapes_match(a, b),         OpType::MatMul => {
    //             if a.layout().shape().len() < 2 || b.layout().shape().len() < 2 {
    //                 return Err(crate::Error::MatMulInvalidShape {
    //                     a_shape: a.layout().shape().clone(),
    //                     b_shape: b.layout().shape().clone(),
    //                 });
    //             }

    //             let a_last = *a.layout().shape().into_iter().last().unwrap();
    //             let b_second_last =
    // *b.layout().shape().into_iter().rev().nth(1).unwrap();

    //             if a_last != b_second_last {
    //                 return Err(crate::Error::MatMulDimensionMismatch {
    //                     a_last,
    //                     b_second_last,
    //                     a_shape: a.layout().shape().clone(),
    //                     b_shape: b.layout().shape().clone(),
    //                 });
    //             }
    //             Ok(())
    //         }
    //         _ => Ok(()),
    //     }
    // }
}
