use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::{Add, Div, Index, Mul, RangeInclusive, Sub};
use std::sync::{Arc, Mutex};

use crate::autograd::{Node, NodeId, Tape};
use crate::backend::Backend;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::ops::OpType;
use crate::storage::Storage as _;
use crate::Result;

macro_rules! impl_binary_op {
    ($($trait:ident, $method:ident, $op:expr);* $(;)?) => {
        $(
            // 1. Owned implementation: a + b
            impl<T:DType, B: Backend<T>> $trait for Tensor<T, B> {
                type Output = Result<Self>;
                fn $method(self, rhs: Self) -> Self::Output {
                    self.binary_op(&rhs, $op, |backend, a, b| backend.$method(a, b))
                }
            }

            // 2. Reference implementation: &a + &b
            impl<'a, 'b, T:DType, B: Backend<T>> $trait<&'b Tensor<T, B>> for &'a Tensor<T, B> {
                type Output = Result<Tensor<T, B>>;
                fn $method(self, rhs: &'b Tensor<T, B>) -> Self::Output {
                    self.binary_op(rhs, $op, |backend, a, b| backend.$method(a,b))
                }
            }
        )*
    };
}

#[derive(Debug, Clone)]
pub struct Tensor<T: DType, B: Backend<T>> {
    storage:       B::Storage,
    layout:        TensorLayout,
    backend:       Arc<B>,
    requires_grad: bool,
    tape:          Option<Arc<Tape<T, B>>>,
    node_id:       Option<NodeId>,
    grad:          Option<Arc<Mutex<Option<B::Storage>>>>,
}

impl_binary_op!(
    Add, add, OpType::Add;
    Sub, sub, OpType::Sub;
    Mul, mul, OpType::Mul;
    Div, div, OpType::Div;
);

impl<T: DType, B: Backend<T>> Tensor<T, B> {
    pub fn i(&self, indices: &[usize]) -> Result<&T> {
        let idx = self
            .layout
            .ravel_index(indices)
            .expect("invalid index for tensor");
        Ok(&self.storage.i(idx))
    }

    /// Create a new tensor filled with zeros, given a specific layout and
    /// backend.
    pub fn zeros(layout: TensorLayout, backend: Arc<B>) -> Self {
        let storage = backend.store_zeros(&layout);

        Tensor {
            storage,
            layout,
            backend,
            requires_grad: false,
            tape: None,
            node_id: None,
            grad: None,
        }
    }

    /// Create a new tensor filled with ones, given a specific layout and
    /// backend.
    pub fn ones(layout: TensorLayout, backend: Arc<B>) -> Self {
        let storage = backend.store_ones(&layout);

        Tensor {
            storage,
            layout,
            backend,
            requires_grad: false,
            tape: None,
            node_id: None,
            grad: None,
        }
    }

    pub fn requires_grad(mut self, tape: Arc<Tape<T, B>>) -> Self {
        let grad_slot = Arc::new(Mutex::new(None));
        let node_id = tape.add_node(Node::new(
            OpType::Leaf,
            Vec::new(),
            true,
            self.layout.clone(),
            self.storage.clone(),
            Some(grad_slot.clone()),
        ));

        self.requires_grad = true;
        self.tape = Some(tape);
        self.node_id = Some(node_id);
        self.grad = Some(grad_slot);
        self
    }

    /// Get a reference to the tensor's layout.
    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    /// Get a reference to the tensor's storage.
    pub fn storage(&self) -> &B::Storage {
        &self.storage
    }

    /// Get a reference to the tensor's backend.
    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    pub fn from_parts(storage: B::Storage, layout: TensorLayout, backend: Arc<B>) -> Self {
        Tensor {
            storage,
            layout,
            backend,
            requires_grad: false,
            tape: None,
            node_id: None,
            grad: None,
        }
    }

    fn binary_op<F>(&self, rhs: &Tensor<T, B>, op: OpType, f: F) -> Result<Self>
    where
        F: FnOnce(&B, &Tensor<T, B>, &Tensor<T, B>) -> Result<Tensor<T, B>>,
    {
        op.validate_binop(self, rhs)?;

        let mut out = f(&self.backend, self, rhs)?;

        let needs_grad = self.requires_grad || rhs.requires_grad;
        out.requires_grad = needs_grad;

        if !needs_grad {
            return Ok(out);
        }

        let tape = match (&self.tape, &rhs.tape) {
            (Some(left), Some(right)) if Arc::ptr_eq(left, right) => Some(left.clone()),
            (Some(left), None) => Some(left.clone()),
            (None, Some(right)) => Some(right.clone()),
            _ => None,
        };

        if let Some(tape) = tape {
            let left_id = self.node_id.unwrap_or_else(|| {
                tape.add_node(Node::new(
                    OpType::Leaf,
                    Vec::new(),
                    self.requires_grad,
                    self.layout.clone(),
                    self.storage.clone(),
                    self.grad.clone(),
                ))
            });

            let right_id = rhs.node_id.unwrap_or_else(|| {
                tape.add_node(Node::new(
                    OpType::Leaf,
                    Vec::new(),
                    self.requires_grad,
                    self.layout.clone(),
                    self.storage.clone(),
                    self.grad.clone(),
                ))
            });

            let out_grad = Arc::new(Mutex::new(None));
            let out_id = tape.add_node(Node::new(
                op,
                vec![left_id, right_id],
                needs_grad,
                out.layout.clone(),
                out.storage.clone(),
                Some(out_grad.clone()),
            ));

            out.tape = Some(tape);
            out.node_id = Some(out_id);
            out.grad = Some(out_grad);
        }

        Ok(out)
    }

    pub fn backward(&self) -> Result<()> {
        if !self.requires_grad {
            return Ok(());
        }

        let (Some(tape), Some(root_id)) = (&self.tape, self.node_id) else {
            return Ok(());
        };

        let mut stack = vec![root_id];
        let mut visited = HashSet::new();
        let mut order = Vec::new();

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }

            order.push(id);
            if let Some(node) = tape.node(id) {
                for parent in node.parents() {
                    stack.push(*parent);
                }
            }
        }

        order.sort_by_key(|id| id.0);
        order.reverse();

        let mut grads: HashMap<NodeId, Tensor<T, B>> = HashMap::new();
        grads.insert(
            root_id,
            Tensor::ones(self.layout.clone(), self.backend.clone()),
        );

        let add_grad =
            |grads: &mut HashMap<NodeId, Tensor<T, B>>, id: NodeId, grad: Tensor<T, B>| {
                if let Some(existing) = grads.remove(&id) {
                    let sum = (&existing + &grad).expect("tensor addition failed in backward");
                    grads.insert(id, sum);
                } else {
                    grads.insert(id, grad);
                }
            };

        for id in order {
            let grad = match grads.remove(&id) {
                Some(g) => g,
                None => continue,
            };

            let node = match tape.node(id) {
                Some(n) => n,
                None => continue,
            };

            tape.set_grad(id, grad.storage.clone());

            match node.op() {
                OpType::Leaf => continue,
                OpType::Add | OpType::Sub | OpType::Mul | OpType::Div | OpType::MatMul => {
                    if node.parents().len() != 2 {
                        continue;
                    }
                }
            }

            let left = match tape.node(node.parents()[0]) {
                Some(n) => n,
                None => continue,
            };

            let right = match tape.node(node.parents()[1]) {
                Some(n) => n,
                None => continue,
            };

            let a = Tensor::from_parts(
                left.value().clone(),
                left.layout().clone(),
                self.backend.clone(),
            );

            let b = Tensor::from_parts(
                right.value().clone(),
                right.layout().clone(),
                self.backend.clone(),
            );

            match node.op() {
                OpType::Add => {
                    add_grad(&mut grads, node.parents()[0], grad.clone());
                    add_grad(&mut grads, node.parents()[1], grad);
                }
                OpType::Sub => {
                    add_grad(&mut grads, node.parents()[0], grad.clone());
                    let neg = (Tensor::zeros(node.layout().clone(), self.backend.clone()) - grad)
                        .expect("backward subtraction: negation failed");
                    add_grad(&mut grads, node.parents()[1], neg);
                }
                OpType::Mul => {
                    let ga = (&grad * &b).expect("backward multiplication: left grad failed");
                    let gb = (&grad * &a).expect("backward multiplication: right grad failed");
                    add_grad(&mut grads, node.parents()[0], ga);
                    add_grad(&mut grads, node.parents()[1], gb);
                }
                OpType::Div => {
                    let ga = (&grad / &b).expect("backward division: left grad failed");
                    let b_sq = (&b * &b).expect("backward division: b squared failed");
                    let gb = (&grad * &a).expect("backward division: right grad failed");
                    let gb = (&gb / &b_sq).expect("backward division: right grad div failed");
                    let neg = (Tensor::zeros(node.layout().clone(), self.backend.clone()) - gb)
                        .expect("backward division: right grad negation failed");
                    add_grad(&mut grads, node.parents()[0], ga);
                    add_grad(&mut grads, node.parents()[1], neg);
                }
                OpType::MatMul => {
                    // TODO
                }
                OpType::Leaf => {}
            }
        }

        Ok(())
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
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
            grad: self.grad.clone(),
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
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
            grad: self.grad.clone(),
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
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
            grad: self.grad.clone(),
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
                requires_grad: self.requires_grad,
                tape: self.tape.clone(),
                node_id: self.node_id,
                grad: self.grad.clone(),
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
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
            grad: self.grad.clone(),
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
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
            grad: self.grad.clone(),
        })
    }
}

// impl<T: DType, B: Backend<T>> Index<&[usize]> for Tensor<T, B> {
//     type Output = T;

//     fn index(&self, indices: &[usize]) -> &Self::Output {
//         let idx = self
//             .layout
//             .ravel_index(indices)
//             .expect("invalid index for tensor");
//         &self.storage[idx]
//     }
// }
