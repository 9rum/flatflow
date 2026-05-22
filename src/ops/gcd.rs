// SPDX-License-Identifier: Apache-2.0

//! The greatest common divisor utility.

/// Computes the [greatest common divisor] of the integers `m` and `n`.
///
/// # Current implementation
///
/// This implementation employs the [binary GCD algorithm], also known as Stein's algorithm, which
/// replaces division with bitwise shifts and subtractions.
/// See Knuth, [The Art of Computer Programming: Volume 2, Third edition, 1997]
/// Chapter 4.5.2, Algorithm B: Binary gcd algorithm.
///
/// [greatest common divisor]: https://en.wikipedia.org/wiki/greatest_common_divisor
/// [binary GCD algorithm]: https://doi.org/10.1016/0021-9991(67)90047-2
/// [The Art of Computer Programming: Volume 2, Third edition, 1997]: https://dl.acm.org/doi/10.5555/270146
#[inline]
pub(super) const fn gcd(m: i64, n: i64) -> u64 {
    let mut m = m.unsigned_abs();
    let mut n = n.unsigned_abs();

    if m == 0 || n == 0 {
        m | n
    } else {
        let shift = (m | n).trailing_zeros();
        m >>= m.trailing_zeros();
        n >>= n.trailing_zeros();

        while m != n {
            if m < n {
                n -= m;
                n >>= n.trailing_zeros();
            } else {
                m -= n;
                m >>= m.trailing_zeros();
            }
        }

        m << shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd_is_commutative() {
        assert_eq!(gcd(12, 18), gcd(18, 12));
        assert_eq!(gcd(6, 10), gcd(10, 6));
        assert_eq!(gcd(6, -10), gcd(-10, 6));
        assert_eq!(gcd(-6, -10), gcd(-10, -6));
        assert_eq!(gcd(24, 0), gcd(0, 24));
        assert_eq!(gcd(-24, 0), gcd(0, -24));
    }

    #[test]
    fn test_gcd_with_zero_returns_the_other() {
        assert_eq!(gcd(12, 0), 12);
        assert_eq!(gcd(18, 0), 18);
        assert_eq!(gcd(6, 0), 6);
        assert_eq!(gcd(-10, 0), 10);
        assert_eq!(gcd(-24, 0), 24);

        assert_eq!(gcd(i64::MIN, 0), i64::MIN.unsigned_abs());
        assert_eq!(gcd(-1, 0), 1);
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(i64::MAX, 0), i64::MAX.unsigned_abs());
    }

    #[test]
    fn test_gcd_with_one_returns_one() {
        assert_eq!(gcd(12, 1), 1);
        assert_eq!(gcd(18, 1), 1);
        assert_eq!(gcd(6, 1), 1);
        assert_eq!(gcd(-10, 1), 1);
        assert_eq!(gcd(-24, 1), 1);

        assert_eq!(gcd(i64::MIN, 1), 1);
        assert_eq!(gcd(-1, 1), 1);
        assert_eq!(gcd(0, 1), 1);
        assert_eq!(gcd(1, 1), 1);
        assert_eq!(gcd(i64::MAX, 1), 1);
    }

    #[test]
    fn test_gcd_with_boundary_values() {
        assert_eq!(gcd(i64::MIN, i64::MIN), i64::MIN.unsigned_abs());
        assert_eq!(gcd(i64::MIN, i64::MAX), 1);
        assert_eq!(gcd(i64::MAX, i64::MAX), i64::MAX.unsigned_abs());
    }
}
