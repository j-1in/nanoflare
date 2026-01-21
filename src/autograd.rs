use std::sync::{Arc, Mutex};

use crate::backend::Backend;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::ops::OpType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
pub struct Node<T: DType, B: Backend<T>> {
    op:            OpType,
    parents:       Vec<NodeId>,
    requires_grad: bool,
    layout:        TensorLayout,
    value:         B::Storage,
    grad_slot:     Option<Arc<Mutex<Option<B::Storage>>>>,
}

impl<T: DType, B: Backend<T>> Node<T, B> {
    pub fn new(
        op: OpType,
        parents: Vec<NodeId>,
        requires_grad: bool,
        layout: TensorLayout,
        value: B::Storage,
        grad_slot: Option<Arc<Mutex<Option<B::Storage>>>>,
    ) -> Self {
        Node {
            op,
            parents,
            requires_grad,
            layout,
            value,
            grad_slot,
        }
    }

    pub fn parents(&self) -> &Vec<NodeId> {
        &self.parents
    }

    pub fn op(&self) -> OpType {
        self.op
    }

    pub fn value(&self) -> &B::Storage {
        &self.value
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }
}

#[derive(Debug, Default)]
pub struct Tape<T: DType, B: Backend<T>> {
    nodes: Mutex<Vec<Node<T, B>>>,
}

impl<T: DType, B: Backend<T>> Tape<T, B> {
    pub fn new() -> Self {
        Tape { nodes: Mutex::new(Vec::new()) }
    }

    pub fn add_node(&self, node: Node<T, B>) -> NodeId {
        let mut nodes = self.nodes.lock().expect("autograd tape mutex poisoned");
        let id = NodeId(nodes.len());
        nodes.push(node);
        id
    }

    pub fn node(&self, id: NodeId) -> Option<Node<T, B>> {
        self.nodes
            .lock()
            .ok()
            .and_then(|nodes| nodes.get(id.0).cloned())
    }

    pub fn set_grad(&self, id: NodeId, grad: B::Storage) {
        let slot = self
            .nodes
            .lock()
            .ok()
            .and_then(|nodes| nodes.get(id.0).and_then(|node| node.grad_slot.clone()));

        if let Some(slot) = slot {
            if let Ok(mut guard) = slot.lock() {
                *guard = Some(grad);
            }
        }
    }

    pub fn grad(&self, id: NodeId) -> Option<B::Storage> {
        let slot = self
            .nodes
            .lock()
            .ok()
            .and_then(|nodes| nodes.get(id.0).and_then(|node| node.grad_slot.clone()));

        slot.and_then(|slot| slot.lock().ok().and_then(|g| g.clone()))
    }

    pub fn len(&self) -> usize {
        self.nodes.lock().map(|nodes| nodes.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
