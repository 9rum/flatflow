// SPDX-License-Identifier: Apache-2.0

//! Partitioning algorithms for [identical-machines scheduling].
//!
//! [identical-machines scheduling]: https://doi.org/10.1137/0117039

use core::cmp::Ordering;
use core::iter::{empty, repeat, repeat_with};
use core::ops::{Add, Sub};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BinaryHeap, LinkedList};

/// Auxiliary structure for the balanced largest differencing method and the Meld algorithm.
struct Tuple<K, V> {
    subsets: BTreeMap<K, Vec<LinkedList<V>>>,
    spread: K,
}

impl<K, V> Tuple<K, V> {
    /// Constructs a new k-tuple from the given iterable `iter`. The subsets are sorted within the
    /// underlying associative container, according to the order of their subset sums obtained via
    /// projection `f`.
    #[inline]
    fn new<I, F>(iter: I, f: F) -> Self
    where
        I: IntoIterator<Item = V>,
        F: Fn(&V) -> K,
        K: Copy + Ord + Sub<Output = K>,
    {
        let mut subsets: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for item in iter {
            let mut subset = LinkedList::new();
            subset.push_back(item);
            subsets.entry(f(subset.back().unwrap())).or_default().push(subset);
        }

        let spread = *subsets.keys().next_back().unwrap() - *subsets.keys().next().unwrap();

        Self { subsets, spread }
    }

    /// Differences the two given k-tuples by joining the subset with the smallest sum in `self`
    /// with the subset with the largest sum in `other`, the subset with the second smallest sum in
    /// `self` with the subset with the second largest sum in `other`, and so on.
    #[inline]
    fn fold(mut self, mut other: Self) -> Self
    where
        K: Add<Output = K> + Copy + Ord + Sub<Output = K>,
    {
        let mut subsets: BTreeMap<_, Vec<_>> = BTreeMap::new();

        while let Some((min, mut first)) = self.pop_first() {
            let (max, mut last) = other.pop_last().unwrap();
            first.append(&mut last);
            subsets.entry(min + max).or_default().push(first);
        }

        debug_assert!(other.subsets.is_empty());

        let spread = *subsets.keys().next_back().unwrap() - *subsets.keys().next().unwrap();

        Self { subsets, spread }
    }

    /// Fuses the two given k-tuples so that the produced k-tuple has an interim spread large enough
    /// to offset the excessive spread in another k-tuple.
    #[inline]
    fn meld(mut self, other: Self, threshold: K) -> Self
    where
        K: Add<Output = K> + Copy + Ord + Sub<Output = K>,
    {
        // The melding procedure starts with a 2k-tuple merged from the two given k-tuples.
        //
        // Note that [BTreeMap::merge] is not stable as of rustc 1.96.0 so here we iteratively move
        // elements from `other` into `self`.
        //
        // [BTreeMap::merge]: https://doc.rust-lang.org/alloc/collections/btree_map/struct.BTreeMap.html
        for (sum, mut subsets) in other.subsets {
            self.subsets.entry(sum).or_default().append(&mut subsets);
        }

        let mut subsets: BTreeMap<_, Vec<_>> = BTreeMap::new();

        while let Some((min, mut first)) = self.pop_first() {
            let (max, mut last) = self.pop_last().unwrap();

            if self.subsets.is_empty() {
                // If there are only two subsets left in the tuple, simply join the subsets and
                // insert into the tuple.
                first.append(&mut last);
                subsets.entry(min + max).or_default().push(first);
            } else {
                // Else then there are at least four subsets in the tuple. Here a heuristic search
                // is adopted to avoid exhaustive search of *O*(*n^4*), finding a pair of subsets to
                // meet the condition (v_i + v_j ) − (v_l + v_m) ≈ δ(p_1) − δ^− by scanning subsets
                // from the two extreme ends.
                let iter = self.subsets.iter().flat_map(|(k, v)| repeat(k).take(v.len()));
                let rev = self.subsets.iter().rev().flat_map(|(k, v)| repeat(k).take(v.len()));

                let (&left, &right) = iter
                    .zip(rev)
                    .find(|&(&left, &right)| threshold <= max + left - min - right)
                    .unwrap_or_else(|| {
                        (
                            self.subsets.keys().next_back().unwrap(),
                            self.subsets.keys().next().unwrap(),
                        )
                    });

                match self.subsets.entry(left) {
                    Entry::Occupied(mut entry) => {
                        last.append(&mut entry.get_mut().pop().unwrap());
                        if entry.get().is_empty() {
                            entry.remove();
                        }
                        subsets.entry(max + left).or_default().push(last);
                    }
                    Entry::Vacant(_) => unreachable!(),
                }

                match self.subsets.entry(right) {
                    Entry::Occupied(mut entry) => {
                        first.append(&mut entry.get_mut().pop().unwrap());
                        if entry.get().is_empty() {
                            entry.remove();
                        }
                        subsets.entry(min + right).or_default().push(first);
                    }
                    Entry::Vacant(_) => unreachable!(),
                }

                // FIXME: Here the line 14 of algorithm 2 is omitted; each iteration has to reduce
                // δ(p_1) by δ^−.
            }
        }

        let spread = *subsets.keys().next_back().unwrap() - *subsets.keys().next().unwrap();

        Self { subsets, spread }
    }

    #[inline]
    fn pop_first(&mut self) -> Option<(K, LinkedList<V>)>
    where
        K: Copy + Ord,
    {
        let mut entry = self.subsets.first_entry()?;

        let sum = *entry.key();
        let subset = entry.get_mut().pop().unwrap();

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
        let subset = entry.get_mut().pop().unwrap();

        if entry.get().is_empty() {
            entry.remove();
        }

        Some((sum, subset))
    }
}

impl<K, V> PartialEq for Tuple<K, V>
where
    K: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.spread == other.spread
    }
}

impl<K, V> Eq for Tuple<K, V> where K: Eq {}

impl<K, V> PartialOrd for Tuple<K, V>
where
    K: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.spread.partial_cmp(&other.spread)
    }
}

impl<K, V> Ord for Tuple<K, V>
where
    K: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.spread.cmp(&other.spread)
    }
}

/// Partitions the items in the given iterable `iter` into `k` subsets using the balanced largest
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
fn bldm<I, F, K, B>(iter: I, k: usize, f: F) -> B
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    F: Fn(&I::Item) -> K,
    K: Add<Output = K> + Copy + Ord + Sub<Output = K>,
    B: IntoIterator + FromIterator<B::Item>,
    B::Item: FromIterator<I::Item>,
{
    let mut iter = iter.into_iter();

    // BLDM starts with a sequence of k-tuples, each of which is obtained from the `k` smallest
    // remaining items.
    let mut heap = BinaryHeap::with_capacity(iter.len() / k);

    while 0 < iter.len() {
        heap.push(Tuple::new(iter.by_ref().take(k), &f));
    }

    // Next, the algorithm selects two k-tuples from the sequence, for which the difference between
    // the maximum and minimum subset sum is largest. These two k-tuples are combined into a new
    // k-tuple by joining the subset with the smallest sum in one k-tuple with the subset with the
    // largest sum in another k-tuple, the subset with the second smallest sum in one k-tuple with
    // the subset with the second largest sum in another k-tuple, and so on. This process is called
    // differencing the k-tuples. The combined k-tuple replaces the two k-tuples in the sequence and
    // we iterate this differencing operation until only one k-tuple in the sequence remains, which
    // is the final tuple produced by BLDM.
    while 1 < heap.len() {
        let tuple = heap.pop().unwrap().fold(heap.pop().unwrap());
        heap.push(tuple);
    }

    heap.pop()
        .unwrap()
        .subsets
        .into_values()
        .flat_map(|subsets| subsets.into_iter().map(|subset| subset.into_iter().collect()))
        .collect()
}

/// Partitions the items in the given iterable `iter` into `k` subsets using the Meld algorithm from
/// the paper [Heuristic Algorithms for Balanced Multi-Way Number Partitioning], which is designed
/// for skewed data where BLDM falls short.
///
/// [Heuristic Algorithms for Balanced Multi-Way Number Partitioning]: https://www.ijcai.org/Proceedings/11/Papers/122.pdf
#[inline]
fn meld<I, F, K, B>(iter: I, k: usize, f: F) -> B
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    F: Fn(&I::Item) -> K,
    K: Add<Output = K> + Copy + Ord + Sub<Output = K>,
    B: IntoIterator + FromIterator<B::Item>,
    B::Item: FromIterator<I::Item>,
{
    let mut iter = iter.into_iter();

    let mut heap = BinaryHeap::with_capacity(iter.len() / k);

    while 0 < iter.len() {
        heap.push(Tuple::new(iter.by_ref().take(k), &f));
    }

    while 1 < heap.len() {
        let first = heap.pop().unwrap();
        let second = heap.pop().unwrap();

        match heap.peek() {
            // Meld deviates from BLDM’s principle of eliminating the spread whenever a pair of
            // k-tuples is folded. When a k-tuple with excessive spread is found, the algorithm
            // melds the next two k-tuples so that the produced spread counterbalances the largest
            // spread.
            Some(third) if second.spread + third.spread < first.spread => {
                let tuple = second.meld(heap.pop().unwrap(), first.spread);
                heap.push(first.fold(tuple));
            }
            _ => heap.push(first.fold(second)),
        }
    }

    heap.pop()
        .unwrap()
        .subsets
        .into_values()
        .flat_map(|subsets| subsets.into_iter().map(|subset| subset.into_iter().collect()))
        .collect()
}

/// An option to select the approximate algorithm to use for balanced multi-way number partitioning.
#[derive(Clone, Copy, Debug, Default)]
pub enum Heuristic {
    BLDM,
    #[default]
    Meld,
}

/// Reorders the items in the given iterable `iter` into `k` subsets with respect to the given
/// projection `f`. The items in `iter` should be sorted according to the projection `f`, whether in
/// ascending or descending order. The resulting subsets are sorted in ascending order of their
/// subset sums according to the projection `f`.
///
/// # Panics
///
/// Panics if `iter` is not empty but `k` is zero or if `k` does not divide the length of `iter`.
///
/// # Current implementation
///
/// The current algorithm adopts [Meld] by default, an approximate algorithm tailored for skewed
/// distributions. Since this algorithm has time complexity of *O*(*n^2*), the [balanced largest
/// differencing method] (BLDM) can be used by setting `heuristic` to `Some(Heuristic::BLDM)` when
/// the cardinality is sufficiently large or fast search is required. Note that the time complexity
/// of BLDM is *O*(*n log n*). If `heuristic` is set to `None` or `Some(Heuristic::Meld)`, Meld is
/// used for the partitioning.
///
/// There is another approximate algorithm for balanced multi-way number partitioning; LRM by Zhang,
/// Mouratidis and Pang from the paper [Heuristic Algorithms for Balanced Multi-Way Number
/// Partitioning] is designed for uniform distributions with odd cardinality.
///
/// [Meld]: https://www.ijcai.org/Proceedings/11/Papers/122.pdf
/// [balanced largest differencing method]: https://www.jstor.org/stable/3690207
/// [Heuristic Algorithms for Balanced Multi-Way Number Partitioning]: https://www.ijcai.org/Proceedings/11/Papers/122.pdf
#[inline]
pub fn partition<I, F, K, B>(iter: I, k: usize, f: F, heuristic: Option<Heuristic>) -> B
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
        0 => repeat_with(|| empty().collect()).take(k).collect(),
        n => {
            assert_ne!(k, 0);
            assert_eq!(n % k, 0);
            match heuristic.unwrap_or_default() {
                Heuristic::BLDM => bldm(iter, k, f),
                Heuristic::Meld => meld(iter, k, f),
            }
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
        let mut sizes = Vec::with_capacity(65536);
        for size in 1..8193 {
            sizes.extend([size; 8]);
        }

        let subsets: Vec<Vec<_>> = bldm(sizes, 1024, |&size| size);
        assert_eq!(subsets.len(), 1024);
        subsets.iter().for_each(|subset| assert_eq!(subset.len(), 64));

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
            .take(65536)
            .collect();
        sizes.sort_unstable();

        let subsets: Vec<Vec<_>> = bldm(sizes, 1024, |&size| size);
        assert_eq!(subsets.len(), 1024);
        subsets.iter().for_each(|subset| assert_eq!(subset.len(), 64));

        let sums: Vec<usize> = subsets.into_iter().map(|subset| subset.into_iter().sum()).collect();
        assert!(sums.is_sorted());

        let min = *sums.first().unwrap();
        let max = *sums.last().unwrap();
        let spread = (max - min) as f64 / min as f64;
        println!("spread: {spread} (min: {min} max: {max})");
    }

    #[test]
    fn test_meld_with_lognormal_distribution() {
        const MU: f32 = 595.2844634189998;
        const SIGMA: f32 = 952.6487919361658;

        let mut rng = StdRng::seed_from_u64(0);
        let mut sizes: Vec<_> = LogNormal::new(MU, SIGMA)
            .unwrap()
            .sample_iter(&mut rng)
            .filter_map(|size| {
                if 0.5 <= size && size < 8192.5 { Some(size.round() as usize) } else { None }
            })
            .take(65536)
            .collect();
        sizes.sort_unstable();

        let subsets: Vec<Vec<_>> = meld(sizes, 1024, |&size| size);
        assert_eq!(subsets.len(), 1024);
        subsets.iter().for_each(|subset| assert_eq!(subset.len(), 64));

        let sums: Vec<usize> = subsets.into_iter().map(|subset| subset.into_iter().sum()).collect();
        assert!(sums.is_sorted());

        let min = *sums.first().unwrap();
        let max = *sums.last().unwrap();
        let spread = (max - min) as f64 / min as f64;
        println!("spread: {spread} (min: {min} max: {max})");
    }

    #[test]
    fn test_partition_with_empty_items() {
        let subsets: Vec<Vec<usize>> = partition([], 0, |&size| size, None);
        assert!(subsets.is_empty());

        let subsets: Vec<Vec<usize>> = partition([], 1024, |&size| size, None);
        assert_eq!(subsets.len(), 1024);
        subsets.into_iter().for_each(|subset| assert!(subset.is_empty()));
    }
}
