// SPDX-License-Identifier: Apache-2.0

//! Structures and associated conversions for serialized [FX graph]s.
//!
//! This module defines representations for the computational graph such as symbolic integers,
//! tensor metadata and nodes, translating them from the generated types in the `*_generated`
//! modules.
//!
//! [FX graph]: https://docs.pytorch.org/docs/stable/fx.html

use flatbuffers::{
    ForwardsUOffset, InvalidFlatbuffer, SIZE_UOFFSET, UOffsetT, Vector, read_scalar, read_scalar_at,
};
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::ops::graph_generated::{self, root_as_graph};
use crate::ops::operator_generated::Operator;
use crate::ops::scalar_type_generated::ScalarType;

/// `SymInt` records a value within the symbolic shape of a tensor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SymInt(pub i64, pub i64);

impl From<&graph_generated::SymInt> for SymInt {
    #[inline]
    fn from(int: &graph_generated::SymInt) -> Self {
        Self(int.data().get(0), int.data().get(1))
    }
}

/// `TensorMetadata` is a structure containing pertinent information about a tensor within a PyTorch
/// program.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TensorMetadata {
    pub dtype: ScalarType,
    pub shape: Vec<SymInt>,
}

impl From<graph_generated::TensorMetadata<'_>> for TensorMetadata {
    #[inline]
    fn from(meta: graph_generated::TensorMetadata<'_>) -> Self {
        Self { dtype: meta.dtype(), shape: meta.shape().iter().map(Into::into).collect() }
    }
}

/// `Node` is a data structure that represents individual operations in the computational graph.
/// Each node contains an opcode identifying operators and the input/output shapes of the operator.
/// Unlike `torch.fx.Node`, this excludes operations other than callsites to ATen operators; i.e.,
/// operations whose `op` property are not `call_function`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Node {
    pub target: Operator,
    pub args: Vec<TensorMetadata>,
    pub meta: TensorMetadata,
}

impl From<graph_generated::Node<'_>> for Node {
    #[inline]
    fn from(node: graph_generated::Node<'_>) -> Self {
        Self {
            target: node.target(),
            args: node.args().iter().map(Into::into).collect(),
            meta: node.meta().into(),
        }
    }
}

/// `Graph` is the main data structure for tracing a given PyTorch program at the intermediate
/// representation level. It consists of a series of `Node`s, each representing callsites such as
/// opcode and the input/output shapes of the corresponding operator.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

/// Parallel iterator over a `Graph`.
#[derive(Clone, Copy, Debug)]
pub struct GraphIter<'a> {
    buf: &'a [u8],
    loc: usize,
    len: usize,
}

impl<'a, 'b> From<&'b graph_generated::Graph<'a>> for GraphIter<'a> {
    #[inline]
    fn from(graph: &'b graph_generated::Graph<'a>) -> Self {
        let loc =
            graph._tab.loc() + graph._tab.vtable().get(graph_generated::Graph::VT_NODES) as usize;
        let slice = &graph._tab.buf()[loc..loc + SIZE_UOFFSET];
        let offset = unsafe { read_scalar::<UOffsetT>(slice) } as usize;

        Self {
            buf: graph._tab.buf(),
            loc: loc + offset + SIZE_UOFFSET,
            len: unsafe { read_scalar_at::<UOffsetT>(graph._tab.buf(), loc + offset) } as usize,
        }
    }
}

impl<'a> Producer for GraphIter<'a> {
    type Item = <Vector<'a, ForwardsUOffset<graph_generated::Node<'a>>> as IntoIterator>::Item;
    type IntoIter =
        <Vector<'a, ForwardsUOffset<graph_generated::Node<'a>>> as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        unsafe { Self::IntoIter::from_slice(&self.buf[self.loc..], self.len) }
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        (
            Self { buf: self.buf, loc: self.loc, len: index },
            Self { buf: self.buf, loc: self.loc + SIZE_UOFFSET * index, len: self.len - index },
        )
    }
}

impl<'a> ParallelIterator for GraphIter<'a> {
    type Item = <Self as Producer>::Item;

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.len)
    }
}

impl<'a> IndexedParallelIterator for GraphIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

impl<'a, 'b> IntoParallelIterator for &'b graph_generated::Graph<'a> {
    type Iter = GraphIter<'a>;
    type Item = <Self::Iter as ParallelIterator>::Item;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        self.into()
    }
}

impl From<graph_generated::Graph<'_>> for Graph {
    #[inline]
    fn from(graph: graph_generated::Graph<'_>) -> Self {
        Self { nodes: graph.par_iter().map(Into::into).collect() }
    }
}

impl TryFrom<&[u8]> for Graph {
    type Error = InvalidFlatbuffer;

    #[inline]
    fn try_from(buf: &[u8]) -> Result<Self, Self::Error> {
        Ok(root_as_graph(buf)?.into())
    }
}
