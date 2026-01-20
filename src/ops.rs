use crate::{Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Leaf,
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
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
    pub fn validate_binop<T, B>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()>
    where
        T: crate::dtype::DType,
        B: crate::backend::Backend<T>,
    {
        match self {
            OpType::Add | OpType::Sub | OpType::Mul | OpType::Div => shapes_match(a, b),
            _ => Ok(()),
        }
    }
}
