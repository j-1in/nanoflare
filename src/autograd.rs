use std::sync::{Arc, Mutex};

use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::ops::OpType;
use crate::storage::TensorStorage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
pub struct Node<T: DType> {
    pub op:            OpType,
    pub parents:       Vec<NodeId>,
    pub requires_grad: bool,
    pub layout:        TensorLayout,
    pub value:         TensorStorage<T>,
    pub grad_slot:     Option<Arc<Mutex<Option<TensorStorage<T>>>>>,
}

#[derive(Debug, Default)]
pub struct Tape<T: DType> {
    nodes: Mutex<Vec<Node<T>>>,
}

impl<T: DType> Tape<T> {
    pub fn new() -> Self {
        Tape { nodes: Mutex::new(Vec::new()) }
    }

    pub fn add_node(&self, node: Node<T>) -> NodeId {
        let mut nodes = self.nodes.lock().expect("autograd tape mutex poisoned");
        let id = NodeId(nodes.len());
        nodes.push(node);
        id
    }

    pub fn node(&self, id: NodeId) -> Option<Node<T>> {
        self.nodes
            .lock()
            .ok()
            .and_then(|nodes| nodes.get(id.0).cloned())
    }

    pub fn set_grad(&self, id: NodeId, grad: TensorStorage<T>) {
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

    pub fn grad(&self, id: NodeId) -> Option<TensorStorage<T>> {
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
