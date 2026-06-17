// SPDX-License-Identifier: Apache-2.0

//! Partitioning algorithms for [identical-machines scheduling].
//!
//! [identical-machines scheduling]: https://en.wikipedia.org/wiki/Identical-machines_scheduling

use core::cmp::Ordering;
use core::ops::Sub;
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
