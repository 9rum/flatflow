// SPDX-License-Identifier: Apache-2.0

//! Basic polynomial manipulation functionalities.
//!
//! This module is intended to be used in place of Boost polynomials.

use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign, Sub,
    SubAssign,
};

use crate::ops::gcd::gcd;
use crate::ops::graph::SymInt;

/// This is a trivial structure for polynomial manipulation, as an alternative to Boost polynomials.
/// A notable API difference lies in the absence of division for polynomials over a field and over a
/// unique factorization domain; we soon noticed that implementing symbolic transformations is
/// equivalent to that of polynomial manipulation where the division functionality between
/// polynomials is not required.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Polynomial(pub i64, pub i64, pub i64);

impl Polynomial {
    /// Constructs a new polynomial with the given coefficients.
    #[inline]
    pub const fn new(c0: i64, c1: i64, c2: i64) -> Self {
        Self(c0, c1, c2)
    }

    /// Based on Horner's rule, evaluates a given polynomial of degree two with only two
    /// multiplications and two additions, applying Horner's method.
    /// See Knuth, [The Art of Computer Programming: Volume 2, Third edition, 1997]
    /// Chapter 4.6.4, Horner's rule.
    ///
    /// This is optimal, since there are polynomials of degree two that cannot be evaluated with
    /// fewer arithmetic operations.
    /// See [Methods of computing values of polynomials].
    ///
    /// [The Art of Computer Programming: Volume 2, Third edition, 1997]: https://dl.acm.org/doi/10.5555/270146
    /// [Methods of computing values of polynomials]: https://doi.org/10.1070%2Frm1966v021n01abeh004147
    #[inline]
    pub fn eval<T>(&self, value: T) -> Result<i64, T::Error>
    where
        T: TryInto<i64>,
    {
        Ok(self.eval_impl(value.try_into()?))
    }

    #[inline]
    const fn eval_impl(&self, value: i64) -> i64 {
        self.0 + value * (self.1 + value * self.2)
    }

    /// Normalizes `self` so that the constant term becomes zero and the rest are relatively prime.
    #[inline]
    pub const fn normalize(&mut self) {
        *self = match gcd(self.1, self.2) {
            // This value is |i64::MIN|, which cannot be represented within i64. There are two
            // possible cases for this: both self.1 and self.2 are i64::MIN, or one of them is
            // i64::MIN and the other is zero. In both cases the normalized value is -1 for i64::MIN
            // and 0 for zero, which corresponds to their respective signum.
            0x8000000000000000 => Self(0, self.1.signum(), self.2.signum()),
            // The only case where the divisor is zero is when both self.1 and self.2 are zero, in
            // which case there is no need to divide them.
            0 => Self(0, 0, 0),
            // Otherwise the divisor can be safely cast to i64 and used for division.
            divisor => {
                let divisor = divisor.cast_signed();
                Self(0, self.1 / divisor, self.2 / divisor)
            }
        };
    }
}

impl From<i64> for Polynomial {
    #[inline]
    fn from(c0: i64) -> Self {
        Self(c0, 0, 0)
    }
}

impl From<&SymInt> for Polynomial {
    /// Constructs a new polynomial from the given symbolic integer.
    #[inline]
    fn from(int: &SymInt) -> Self {
        Self(int.0, int.1, 0)
    }
}

/// `polynomial!` allows `Polynomial`s to be defined with an arbitrary number of values.
///
/// Note that the arguments are used to value-initialize the underlying fixed-size tuple, and may
/// not exceed the container capacity.
#[macro_export]
macro_rules! polynomial {
    () => {
        $crate::ops::polynomial::Polynomial(0, 0, 0)
    };
    ($c0:expr) => {
        $crate::ops::polynomial::Polynomial($c0, 0, 0)
    };
    ($c0:expr, $c1:expr) => {
        $crate::ops::polynomial::Polynomial($c0, $c1, 0)
    };
    ($c0:expr, $c1:expr, $c2:expr) => {
        $crate::ops::polynomial::Polynomial($c0, $c1, $c2)
    };
}

macro_rules! offset_op_impl_scalar {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl $trait<i64> for Polynomial {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: i64) -> Self::Output {
                Self(self.0.$method(rhs), self.1, self.2)
            }
        }

        impl $assign_trait<i64> for Polynomial {
            #[inline]
            fn $assign_method(&mut self, rhs: i64) {
                self.0.$assign_method(rhs);
            }
        }
    };
}

offset_op_impl_scalar!(Add, add, AddAssign, add_assign);
offset_op_impl_scalar!(Sub, sub, SubAssign, sub_assign);

impl Add<Polynomial> for i64 {
    type Output = Polynomial;

    #[inline]
    fn add(self, rhs: Polynomial) -> Self::Output {
        rhs + self
    }
}

impl Sub<Polynomial> for i64 {
    type Output = Polynomial;

    #[inline]
    fn sub(self, rhs: Polynomial) -> Self::Output {
        -rhs + self
    }
}

macro_rules! scale_op_impl_scalar {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl $trait<i64> for Polynomial {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: i64) -> Self::Output {
                Self(self.0.$method(rhs), self.1.$method(rhs), self.2.$method(rhs))
            }
        }

        impl $assign_trait<i64> for Polynomial {
            #[inline]
            fn $assign_method(&mut self, rhs: i64) {
                self.0.$assign_method(rhs);
                self.1.$assign_method(rhs);
                self.2.$assign_method(rhs);
            }
        }
    };
}

scale_op_impl_scalar!(Mul, mul, MulAssign, mul_assign);
scale_op_impl_scalar!(Div, div, DivAssign, div_assign);
scale_op_impl_scalar!(Shl, shl, ShlAssign, shl_assign);
scale_op_impl_scalar!(Shr, shr, ShrAssign, shr_assign);

impl Mul<Polynomial> for i64 {
    type Output = Polynomial;

    #[inline]
    fn mul(self, rhs: Polynomial) -> Self::Output {
        rhs * self
    }
}

macro_rules! offset_op_impl_polynomial {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl $trait for Polynomial {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                Self(self.0.$method(rhs.0), self.1.$method(rhs.1), self.2.$method(rhs.2))
            }
        }

        impl $assign_trait for Polynomial {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                self.0.$assign_method(rhs.0);
                self.1.$assign_method(rhs.1);
                self.2.$assign_method(rhs.2);
            }
        }
    };
}

offset_op_impl_polynomial!(Add, add, AddAssign, add_assign);
offset_op_impl_polynomial!(Sub, sub, SubAssign, sub_assign);

impl Mul for Polynomial {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(
            self.0 * rhs.0,
            self.0 * rhs.1 + self.1 * rhs.0,
            self.0 * rhs.2 + self.1 * rhs.1 + self.2 * rhs.0,
        )
    }
}

impl MulAssign for Polynomial {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self(
            self.0 * rhs.0,
            self.0 * rhs.1 + self.1 * rhs.0,
            self.0 * rhs.2 + self.1 * rhs.1 + self.2 * rhs.0,
        )
    }
}

impl Neg for Polynomial {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1, -self.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        assert_eq!(Polynomial::new(1, 2, 3), Polynomial(1, 2, 3));

        assert_eq!(Polynomial::default(), Polynomial(0, 0, 0));
        assert_eq!(Polynomial::default(), Polynomial::new(0, 0, 0));

        assert_eq!(Polynomial::from(2), Polynomial(2, 0, 0));
        assert_eq!(Polynomial::from(3), Polynomial::new(3, 0, 0));
        assert_eq!(Polynomial::from(0), Polynomial::default());

        assert_eq!(polynomial!(), Polynomial::default());
        assert_eq!(polynomial!(1), Polynomial::from(1));
        assert_eq!(polynomial!(1, 2), Polynomial(1, 2, 0));
        assert_eq!(polynomial!(1, 2, 3), Polynomial(1, 2, 3));
    }

    #[test]
    fn test_scalar_arithmetic() {
        let p = polynomial!(12, 18, 30);

        assert_eq!(p + 0, p);
        assert_eq!(0 + p, p);
        assert_eq!(p + 3, polynomial!(15, 18, 30));
        assert_eq!(3 + p, p + 3);
        assert_eq!(p - 0, p);
        assert_eq!(0 - p, -p);
        assert_eq!(p - 3, polynomial!(9, 18, 30));
        assert_eq!(3 - p, polynomial!(-9, -18, -30));

        assert_eq!(p * 1, p);
        assert_eq!(1 * p, p);
        assert_eq!(p * 2, polynomial!(24, 36, 60));
        assert_eq!(2 * p, p * 2);
        assert_eq!(p / 1, p);
        assert_eq!(p / 3, polynomial!(4, 6, 10));

        assert_eq!(p << 0, p);
        assert_eq!(p << 2, polynomial!(48, 72, 120));
        assert_eq!(p >> 0, p);
        assert_eq!(p >> 1, polynomial!(6, 9, 15));

        let mut p = polynomial!(16, 24, 32);

        p += 1;
        assert_eq!(p, polynomial!(17, 24, 32));

        p -= 1;
        assert_eq!(p, polynomial!(16, 24, 32));

        p *= 3;
        assert_eq!(p, polynomial!(48, 72, 96));

        p /= 4;
        assert_eq!(p, polynomial!(12, 18, 24));

        p <<= 2;
        assert_eq!(p, polynomial!(48, 72, 96));

        p >>= 3;
        assert_eq!(p, polynomial!(6, 9, 12));
    }

    #[test]
    fn test_polynomial_arithmetic() {
        let p = polynomial!(3, 2, 1);

        assert_eq!(p + Polynomial::default(), p);
        assert_eq!(p + -p, Polynomial::default());
        assert_eq!(p + polynomial!(1, 2, 0), polynomial!(4, 4, 1));

        assert_eq!(p - Polynomial::default(), p);
        assert_eq!(p - p, Polynomial::default());
        assert_eq!(p - polynomial!(1, 2, 0), polynomial!(2, 0, 1));

        assert_eq!(-Polynomial::default(), Polynomial::default());
        assert_eq!(-p, polynomial!(-3, -2, -1));

        let mut p = polynomial!(16, 24, 32);

        p += polynomial!(6, 9, 15);
        assert_eq!(p, polynomial!(22, 33, 47));

        p -= polynomial!(4, 6, 10);
        assert_eq!(p, polynomial!(18, 27, 37));
    }

    #[test]
    fn test_polynomial_multiplication() {
        let p = polynomial!(5, 8, 13);
        assert_eq!(p * Polynomial::default(), Polynomial::default());
        assert_eq!(p * polynomial!(1), p);
        assert_eq!(p * polynomial!(1, 2), polynomial!(5, 18, 29));
        assert_eq!(p * polynomial!(1, 2, 3), polynomial!(5, 18, 44));
    }

    #[test]
    fn test_normalization() {
        let mut p = polynomial!(5, 10, 20);
        p.normalize();
        assert_eq!(p, polynomial!(0, 1, 2));

        p = Polynomial::default();
        p.normalize();
        assert_eq!(p, Polynomial::default());

        p = polynomial!(-1, -2, -4);
        p.normalize();
        assert_eq!(p, polynomial!(0, -1, -2));

        p = polynomial!(i64::MIN, i64::MIN, i64::MIN);
        p.normalize();
        assert_eq!(p, polynomial!(0, -1, -1));

        p = polynomial!(i64::MIN, i64::MIN);
        p.normalize();
        assert_eq!(p, polynomial!(0, -1));

        p = polynomial!(0, i64::MIN, i64::MAX);
        p.normalize();
        assert_eq!(p, polynomial!(0, i64::MIN, i64::MAX));
    }

    #[test]
    fn test_polynomial_evaluation() {
        let mut p = polynomial!(3, 2, 1);
        p.normalize();
        assert_eq!(p.eval(0), Ok(0));
        assert_eq!(p.eval(1), Ok(3));
        assert_eq!(p.eval(2), Ok(8));

        p = polynomial!(i64::MIN, i64::MIN, i64::MIN);
        p.normalize();
        assert_eq!(p.eval(0), Ok(0));
        assert_eq!(p.eval(1), Ok(-2));
        assert_eq!(p.eval(2), Ok(-6));

        p = polynomial!();
        p.normalize();
        assert_eq!(p.eval(i64::MAX), Ok(0));
    }
}
