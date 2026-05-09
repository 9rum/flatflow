// SPDX-License-Identifier: Apache-2.0

//! The greatest common divisor utility.

use core::ops::{BitOr, Shl, Shr, Sub};

/// Trait for types that return its absolute value without any wrapping or panicking.
pub(super) trait UnsignedAbs {
    type Output;

    fn unsigned_abs(self) -> Self::Output;
}

macro_rules! unsigned_abs_impl {
    ($signed:ty => $unsigned:ty) => {
        impl UnsignedAbs for $signed {
            type Output = $unsigned;

            #[inline]
            fn unsigned_abs(self) -> Self::Output {
                Self::unsigned_abs(self)
            }
        }
    };
}

unsigned_abs_impl!(i8 => u8);
unsigned_abs_impl!(i16 => u16);
unsigned_abs_impl!(i32 => u32);
unsigned_abs_impl!(i64 => u64);
unsigned_abs_impl!(i128 => u128);
unsigned_abs_impl!(isize => usize);

/// Trait for types that return the number of trailing zeros in its binary representation.
pub(super) trait TrailingZeros {
    fn trailing_zeros(self) -> u32;
}

macro_rules! trailing_zeros_impl {
    ($($t:ty),* $(,)?) => {
        $(
            impl TrailingZeros for $t {
                #[inline]
                fn trailing_zeros(self) -> u32 {
                    Self::trailing_zeros(self)
                }
            }
        )*
    };
}

trailing_zeros_impl!(u8, u16, u32, u64, u128, usize);

/// Computes the greatest common divisor of the integers `m` and `n`.
///
/// # Current implementation
///
/// This implementation employs the [binary GCD algorithm], also known as Stein's algorithm, which
/// replaces division with bitwise shifts and subtractions.
/// See Knuth, [The Art of Computer Programming: Volume 2, Third edition, 1997]
/// Chapter 4.5.2, Algorithm B: Binary gcd algorithm.
///
/// [binary GCD algorithm]: https://doi.org/10.1016/0021-9991(67)90047-2
/// [The Art of Computer Programming: Volume 2, Third edition, 1997]: https://dl.acm.org/doi/10.5555/270146
#[inline]
pub(super) fn gcd<T>(m: T, n: T) -> T::Output
where
    T: UnsignedAbs,
    T::Output: BitOr<Output = T::Output>
        + Copy
        + Default
        + PartialOrd
        + Shl<u32, Output = T::Output>
        + Shr<u32, Output = T::Output>
        + Sub<Output = T::Output>
        + TrailingZeros,
{
    let mut m = m.unsigned_abs();
    let mut n = n.unsigned_abs();

    if m == T::Output::default() || n == T::Output::default() {
        m | n
    } else {
        let shift = (m | n).trailing_zeros();
        m = m >> m.trailing_zeros();
        n = n >> n.trailing_zeros();

        while m != n {
            if m < n {
                n = n - m;
                n = n >> n.trailing_zeros();
            } else {
                m = m - n;
                m = m >> m.trailing_zeros();
            }
        }

        m << shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! gen_test_suite {
        ($name:ident, $t:ty) => {
            mod $name {
                use super::*;

                #[test]
                fn trivial() {
                    assert_eq!(gcd(12 as $t, 18), 6);
                    assert_eq!(gcd(18 as $t, 12), 6);

                    assert_eq!(gcd(6 as $t, 10), 2);
                    assert_eq!(gcd(10 as $t, 6), 2);
                    assert_eq!(gcd(6 as $t, -10), 2);
                    assert_eq!(gcd(-10 as $t, 6), 2);
                    assert_eq!(gcd(-6 as $t, -10), 2);
                    assert_eq!(gcd(-10 as $t, -6), 2);

                    assert_eq!(gcd(24 as $t, 0), 24);
                    assert_eq!(gcd(0 as $t, 24), 24);
                    assert_eq!(gcd(-24 as $t, 0), 24);
                    assert_eq!(gcd(0 as $t, -24), 24);
                }

                #[test]
                fn boundary() {
                    assert_eq!(gcd(0 as $t, 0), 0);
                    assert_eq!(gcd(0 as $t, 1), 1);
                    assert_eq!(gcd(1 as $t, 0), 1);
                    assert_eq!(gcd(-1 as $t, 0), 1);
                    assert_eq!(gcd(0 as $t, -1), 1);

                    assert_eq!(gcd(<$t>::MIN, 0), <$t>::MIN.unsigned_abs());
                    assert_eq!(gcd(0, <$t>::MIN), <$t>::MIN.unsigned_abs());
                    assert_eq!(gcd(<$t>::MIN, 1), 1);
                    assert_eq!(gcd(1, <$t>::MIN), 1);
                    assert_eq!(gcd(<$t>::MIN, 2), 2);
                    assert_eq!(gcd(2, <$t>::MIN), 2);

                    assert_eq!(gcd(<$t>::MIN, <$t>::MIN), <$t>::MIN.unsigned_abs());
                    assert_eq!(gcd(<$t>::MIN, <$t>::MAX), 1);
                    assert_eq!(gcd(<$t>::MAX, <$t>::MIN), 1);

                    assert_eq!(gcd(<$t>::MAX, 0), <$t>::MAX.cast_unsigned());
                    assert_eq!(gcd(0, <$t>::MAX), <$t>::MAX.cast_unsigned());
                    assert_eq!(gcd(<$t>::MAX, <$t>::MAX), <$t>::MAX.cast_unsigned());
                }
            }
        };
    }

    gen_test_suite!(i8, i8);
    gen_test_suite!(i16, i16);
    gen_test_suite!(i32, i32);
    gen_test_suite!(i64, i64);
    gen_test_suite!(i128, i128);
    gen_test_suite!(isize, isize);
}
