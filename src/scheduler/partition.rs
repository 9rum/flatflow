// SPDX-License-Identifier: Apache-2.0

//! Partitioning algorithms for [identical-machines scheduling].
//!
//! [identical-machines scheduling]: https://en.wikipedia.org/wiki/Identical-machines_scheduling

use core::cmp::Ordering;
use core::iter::{empty, repeat_with};
use core::ops::{Add, Sub};
use std::collections::{BTreeMap, BinaryHeap, LinkedList, VecDeque};

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

/// Partitions the items in the given iterable `iter` into `m` subsets using the balanced largest
/// differencing method (BLDM) from the paper [The Differencing Algorithm LDM for Partitioning: A
/// Proof of a Conjecture of Karmarkar and Karp], a variant of LDM for balanced number partitioning
/// with larger cardinalities.
///
/// Note: Description of the algorithm and the proof of performance ratios are provided by Michiels,
/// Aarts, Korst, van Leeuwen and Spieksma from the paper [Computer-assisted proof of performance
/// ratios for the Differencing Method].
///
/// [The Differencing Algorithm LDM for Partitioning: A Proof of a Conjecture of Karmarkar and Karp]: https://www.jstor.org/stable/3690207
/// [Computer-assisted proof of performance ratios for the Differencing Method]: https://doi.org/10.1016/j.disopt.2011.10.001
#[inline]
fn bldm<I, F, K, B>(iter: I, m: usize, f: F) -> B
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    F: Fn(&I::Item) -> K,
    K: Add<Output = K> + Copy + Ord + Sub<Output = K>,
    B: IntoIterator + FromIterator<B::Item>,
    B::Item: FromIterator<I::Item>,
{
    let mut iter = iter.into_iter();

    // Initially, BLDM starts with a sequence of `k` partial solutions, where each partial solution
    // is obtained from the `m` smallest remaining items.
    let mut heap = BinaryHeap::with_capacity(iter.len() / m);

    while 0 < iter.len() {
        heap.push(Solution::new(iter.by_ref().take(m), &f));
    }

    // Next, the algorithm selects two partial solutions from the sequence, for which the difference
    // between the maximum and minimum subset sum is largest. These two solutions are combined into
    // a new partial solution by joining the subset with the smallest sum in one solution with the
    // subset with the largest sum in another solution, the subset with the second smallest sum in
    // one solution with the subset with the second largest sum in another solution, and so on. This
    // process is called differencing the solutions. The combined solution replaces the two
    // solutions in the sequence, and we iterate this differencing operation until only one solution
    // in the sequence remains, which is the balanced solution obtained by BLDM.
    while 1 < heap.len() {
        let solution = heap.pop().unwrap().difference(heap.pop().unwrap());
        heap.push(solution);
    }

    heap.pop()
        .unwrap()
        .subsets
        .into_values()
        .flat_map(|subsets| subsets.into_iter().map(|subset| subset.into_iter().collect()))
        .collect()
}

/// Reorders the items in the given iterable `iter` into `m` subsets according to the given
/// projection `f`. The items in `iter` must be sorted according to the projection `f`, whether in
/// ascending or descending order. The resulting subsets are sorted in ascending order of their
/// subset sums according to the projection `f`.
///
/// # Panics
///
/// Panics if `iter` is not empty but `m` is zero or if `m` does not divide the length of `iter`.
///
/// # Current implementation
///
/// The current algorithm adopts the [balanced largest differencing method] (BLDM), which may yield
/// partitions with a high work-difference, both when the items are distributed uniformly and when
/// their distribution is skewed. LRM and Meld by Zhang, Mouratidis and Pang from the paper
/// [Heuristic Algorithms for Balanced Multi-Way Number Partitioning] can lower such spread in the
/// respective cases.
///
/// Time complexity: *O*(*n log n*)
///
/// [balanced largest differencing method]: https://www.jstor.org/stable/3690207
/// [Heuristic Algorithms for Balanced Multi-Way Number Partitioning]: https://www.ijcai.org/Proceedings/11/Papers/122.pdf
#[inline]
pub(super) fn partition<I, F, K, B>(iter: I, m: usize, f: F) -> B
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    F: Fn(&I::Item) -> K,
    K: Add<Output = K> + Copy + Ord + Sub<Output = K>,
    B: IntoIterator + FromIterator<B::Item>,
    B::Item: FromIterator<I::Item>,
{
    let iter = iter.into_iter();

    match iter.len() {
        0 => repeat_with(|| empty().collect()).take(m).collect(),
        n => {
            assert_ne!(m, 0);
            assert_eq!(n % m, 0);
            bldm(iter, m, f)
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, LogNormal};

    use super::*;

    #[test]
    fn test_bldm_with_uniform_distribution() {
        let mut sizes = Vec::with_capacity(32768);
        for size in 1..8193 {
            sizes.extend([size; 4]);
        }

        let subsets: Vec<Vec<_>> = bldm(sizes, 4096, |&size| size);
        assert_eq!(subsets.len(), 4096);

        let sums: Vec<usize> = subsets.into_iter().map(|subset| subset.into_iter().sum()).collect();
        assert!(sums.is_sorted());

        let min = *sums.first().unwrap();
        let max = *sums.last().unwrap();
        let spread = (max - min) as f64 / min as f64;
        println!("spread: {spread} (min: {min} max: {max})");
    }

    #[test]
    fn test_bldm_with_lognormal_distribution() {
        const MU: f32 = 595.2844634189998;
        const SIGMA: f32 = 952.6487919361658;

        let mut rng = StdRng::seed_from_u64(0);
        let mut sizes: Vec<_> = LogNormal::new(MU, SIGMA)
            .unwrap()
            .sample_iter(&mut rng)
            .filter_map(|size| {
                if 0.5 <= size && size < 8192.5 { Some(size.round() as usize) } else { None }
            })
            .take(32768)
            .collect();
        sizes.sort();

        let subsets: Vec<Vec<_>> = bldm(sizes, 4096, |&size| size);
        assert_eq!(subsets.len(), 4096);

        let sums: Vec<usize> = subsets.into_iter().map(|subset| subset.into_iter().sum()).collect();
        assert!(sums.is_sorted());

        let min = *sums.first().unwrap();
        let max = *sums.last().unwrap();
        let spread = (max - min) as f64 / min as f64;
        println!("spread: {spread} (min: {min} max: {max})");
    }

    #[test]
    fn test_partition_with_empty_items() {
        let subsets: Vec<Vec<usize>> = partition([], 0, |&size| size);
        assert!(subsets.is_empty());
    }
}
