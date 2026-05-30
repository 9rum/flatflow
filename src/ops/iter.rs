// SPDX-License-Identifier: Apache-2.0

//! Parallel iterator types for [Vector]s.
//!
//! `IntoParallelIterator` and `IntoParallelRefIterator` have been adapted from their Rayon
//! counterparts to bypass the orphan rule.
//!
//! [Vector]: https://docs.rs/flatbuffers/latest/flatbuffers/struct.Vector.html

use core::marker::PhantomData;

use flatbuffers::{Follow, Vector};
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

/// Parallel iterator over a [Vector].
///
/// Unlike FlatBuffers' `VectorIter` which has direct access to the buffer of a `Vector`, this
/// `VectorIter` has no way to access the buffer other than `Vector::bytes`. For this reason, its
/// `buf` points to a slice located `SIZE_UOFFSET` bytes after the data location and `loc` is
/// initialized to zero. On the other hand, `len` is equal to `remaining` of its FlatBuffers
/// counterpart.
///
/// [Vector]: https://docs.rs/flatbuffers/latest/flatbuffers/struct.Vector.html
#[derive(Debug)]
pub(crate) struct VectorIter<'a, T>
where
    T: 'a,
{
    buf: &'a [u8],
    loc: usize,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T> Clone for VectorIter<'a, T>
where
    T: Follow<'a> + 'a,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { buf: self.buf, loc: self.loc, len: self.len, _phantom: PhantomData }
    }
}

impl<'a, T> From<Vector<'a, T>> for VectorIter<'a, T> {
    #[inline]
    fn from(vector: Vector<'a, T>) -> Self {
        Self { buf: vector.bytes(), loc: 0, len: vector.len(), _phantom: PhantomData }
    }
}

impl<'a, 'b, T> From<&'b Vector<'a, T>> for VectorIter<'a, T> {
    #[inline]
    fn from(vector: &'b Vector<'a, T>) -> Self {
        Self { buf: vector.bytes(), loc: 0, len: vector.len(), _phantom: PhantomData }
    }
}

impl<'a, T> ParallelIterator for VectorIter<'a, T>
where
    T: Follow<'a> + 'a + Send,
    T::Inner: Send,
{
    type Item = T::Inner;

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

impl<'a, T> IndexedParallelIterator for VectorIter<'a, T>
where
    T: Follow<'a> + 'a + Send,
    T::Inner: Send,
{
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
        callback.callback(VectorIterProducer::from(self))
    }
}

#[derive(Debug)]
struct VectorIterProducer<'a, T>
where
    T: 'a,
{
    buf: &'a [u8],
    loc: usize,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T> Clone for VectorIterProducer<'a, T>
where
    T: Follow<'a> + 'a,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { buf: self.buf, loc: self.loc, len: self.len, _phantom: PhantomData }
    }
}

impl<'a, T> From<VectorIter<'a, T>> for VectorIterProducer<'a, T> {
    #[inline]
    fn from(iter: VectorIter<'a, T>) -> Self {
        Self { buf: iter.buf, loc: iter.loc, len: iter.len, _phantom: PhantomData }
    }
}

impl<'a, T> Producer for VectorIterProducer<'a, T>
where
    T: Follow<'a> + 'a + Send,
{
    type Item = T::Inner;
    type IntoIter = <Vector<'a, T> as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        unsafe { Self::IntoIter::from_slice(&self.buf[self.loc..], self.len) }
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        (
            Self { buf: self.buf, loc: self.loc, len: index, _phantom: PhantomData },
            Self {
                buf: self.buf,
                loc: self.loc + size_of::<T>() * index,
                len: self.len - index,
                _phantom: PhantomData,
            },
        )
    }
}

pub(crate) trait IntoParallelIterator {
    type Iter: ParallelIterator<Item = Self::Item>;
    type Item: Send;

    fn into_par_iter(self) -> Self::Iter;
}

pub(crate) trait IntoParallelRefIterator<'data> {
    type Iter: ParallelIterator<Item = Self::Item>;
    type Item: Send + 'data;

    fn par_iter(&'data self) -> Self::Iter;
}

impl<'data, I> IntoParallelRefIterator<'data> for I
where
    I: 'data + ?Sized,
    &'data I: IntoParallelIterator,
{
    type Iter = <&'data I as IntoParallelIterator>::Iter;
    type Item = <&'data I as IntoParallelIterator>::Item;

    #[inline]
    fn par_iter(&'data self) -> Self::Iter {
        self.into_par_iter()
    }
}

impl<'a, T> IntoParallelIterator for Vector<'a, T>
where
    T: Follow<'a> + 'a + Send,
    T::Inner: Send,
{
    type Iter = VectorIter<'a, T>;
    type Item = T::Inner;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        self.into()
    }
}

impl<'a, 'b, T> IntoParallelIterator for &'b Vector<'a, T>
where
    T: Follow<'a> + 'a + Send,
    T::Inner: Send,
{
    type Iter = VectorIter<'a, T>;
    type Item = T::Inner;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        self.into()
    }
}
