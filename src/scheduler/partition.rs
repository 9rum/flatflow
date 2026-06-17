// SPDX-License-Identifier: Apache-2.0

//! Partitioning algorithms for [identical-machines scheduling].
//!
//! [identical-machines scheduling]: https://en.wikipedia.org/wiki/Identical-machines_scheduling

use core::cmp::Ordering;
use core::ops::{Add, Sub};
use std::collections::{BTreeMap, LinkedList, VecDeque};

/// Auxiliary structure for the balanced largest differencing method.
struct Solution<K, V> {
    subsets: BTreeMap<K, VecDeque<LinkedList<V>>>,
    delta: K,
}

impl<K, V> Solution<K, V> {
    /// Constructs a new partial solution from the given iterable `iter`. The subsets are sorted
    /// within the underlying associative container, according to the order of their subset sums
    /// obtained via projection `f`.
    #[inline]
    fn new<I, F>(iter: I, f: F) -> Self
    where
        I: IntoIterator<Item = V>,
        F: Fn(&V) -> K,
        K: Copy + Ord + Sub<Output = K>,
    {
        let mut subsets: BTreeMap<_, VecDeque<_>> = BTreeMap::new();
        for item in iter {
            let mut subset = LinkedList::new();
            subset.push_back(item);
            subsets.entry(f(subset.back().unwrap())).or_default().push_back(subset);
        }

        let delta = *subsets.keys().next_back().unwrap() - *subsets.keys().next().unwrap();

        Self { subsets, delta }
    }

    /// Combines the two given solutions by joining the subset with the smallest sum in `self` with
    /// the subset with the largest sum in the `other`, the subset with the second smallest sum in
    /// `self` with the subset with the second largest sum in the `other`, and so on.
    #[inline]
    fn difference(mut self, mut other: Self) -> Self
    where
        K: Add<Output = K> + Copy + Ord + Sub<Output = K>,
    {
        let mut subsets: BTreeMap<_, VecDeque<_>> = BTreeMap::new();

        while let Some((min, mut first)) = self.pop_first() {
            let (max, mut last) = other.pop_last().unwrap();
            first.append(&mut last);
            subsets.entry(min + max).or_default().push_back(first);
        }

        debug_assert!(other.subsets.is_empty());

        let delta = *subsets.keys().next_back().unwrap() - *subsets.keys().next().unwrap();

        Self { subsets, delta }
    }

    #[inline]
    fn pop_first(&mut self) -> Option<(K, LinkedList<V>)>
    where
        K: Copy + Ord,
    {
        let mut entry = self.subsets.first_entry()?;

        let sum = *entry.key();
        let subset = entry.get_mut().pop_front().unwrap();

        if entry.get().is_empty() {
            entry.remove();
        }

        Some((sum, subset))
    }

    #[inline]
    fn pop_last(&mut self) -> Option<(K, LinkedList<V>)>
    where
        K: Copy + Ord,
    {
        let mut entry = self.subsets.last_entry()?;

        let sum = *entry.key();
        let subset = entry.get_mut().pop_back().unwrap();

        if entry.get().is_empty() {
            entry.remove();
        }

        Some((sum, subset))
    }
}

impl<K, V> PartialEq for Solution<K, V>
where
    K: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.delta == other.delta
    }
}

impl<K, V> Eq for Solution<K, V> where K: Eq {}

impl<K, V> PartialOrd for Solution<K, V>
where
    K: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.delta.partial_cmp(&other.delta)
    }
}

impl<K, V> Ord for Solution<K, V>
where
    K: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.delta.cmp(&other.delta)
    }
}
