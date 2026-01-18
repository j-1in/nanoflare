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

pub(crate) fn layouts_match<T, B>(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()>
where
    T: crate::dtype::DType,
    B: crate::backend::Backend<T>,
{
    if a.layout() != b.layout() {
        return Err(crate::Error::LayoutMismatch {
            a: a.layout().clone(),
            b: b.layout().clone(),
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
            OpType::Add | OpType::Sub | OpType::Mul | OpType::Div => layouts_match(a, b),
            _ => Ok(()),
        }
    }
}
