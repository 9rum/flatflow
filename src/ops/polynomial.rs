// SPDX-License-Identifier: Apache-2.0

//! Basic polynomial manipulation functionalities.
//!
//! This module is intended to be used in place of Boost polynomials.

use core::ops::{
    Add, AddAssign, BitOr, Div, DivAssign, Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign,
    Sub, SubAssign,
};

use crate::ops::gcd::{TrailingZeros, UnsignedAbs, gcd};

/// This is a trivial structure for polynomial manipulation, as an alternative to Boost polynomials.
/// A notable API difference lies in the absence of division for polynomials over a field and over a
/// unique factorization domain; we soon noticed that implementing symbolic transformations is
/// equivalent to that of polynomial manipulation where the division functionality between
/// polynomials is not required.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Polynomial<T>(T, T, T);

impl<T> Polynomial<T> {
    /// Constructs a new polynomial with the given coefficients.
    #[inline]
    pub(super) const fn new(c0: T, c1: T, c2: T) -> Self {
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
    pub(super) fn eval<U>(&self, value: U) -> T
    where
        T: Add<Output = T> + Copy + From<U> + Mul<Output = T>,
    {
        self.eval_impl(value.into())
    }

    #[inline]
    fn eval_impl(&self, value: T) -> T
    where
        T: Add<Output = T> + Copy + Mul<Output = T>,
    {
        self.0 + value * (self.1 + value * self.2)
    }

    /// Normalizes `self` so that the constant term becomes zero and the rest are relatively prime.
    #[inline]
    pub(super) fn normalize(&mut self)
    where
        T: Copy
            + Default
            + Div<Output = T>
            + TryFrom<<T as UnsignedAbs>::Output>
            + PartialEq
            + Signum
            + UnsignedAbs,
        <T as UnsignedAbs>::Output: BitOr<Output = <T as UnsignedAbs>::Output>
            + Copy
            + Default
            + PartialOrd
            + Shl<u32, Output = <T as UnsignedAbs>::Output>
            + Shr<u32, Output = <T as UnsignedAbs>::Output>
            + Sub<Output = <T as UnsignedAbs>::Output>
            + TrailingZeros,
    {
        *self = if let Ok(divisor) = gcd(self.1, self.2).try_into() {
            if divisor == T::default() {
                Self::default()
            } else {
                Self(T::default(), self.1 / divisor, self.2 / divisor)
            }
        } else {
            // If `divisor` cannot be represented within `T`, then its actual value is `|T::MIN|`.
            // There are two possible cases for this: both `self.1` and `self.2` are `T::MIN`, or
            // one of them is `T::MIN` and the other is `T::default()`.
            // In both cases the normalized value is `-1` for `T::MIN` and `0` for `T::default()`,
            // which corresponds to their respective signum.
            Self(T::default(), self.1.signum(), self.2.signum())
        };
    }
}

/// Trait for types that return a number representing its sign.
pub(super) trait Signum {
    fn signum(self) -> Self;
}

macro_rules! signum_impl {
    ($($t:ty),* $(,)?) => {
        $(
            impl Signum for $t {
                #[inline]
                fn signum(self) -> Self {
                    Self::signum(self)
                }
            }
        )*
    };
}

signum_impl!(i8, i16, i32, i64, i128, isize);

impl<T> From<T> for Polynomial<T>
where
    T: Default,
{
    #[inline]
    fn from(value: T) -> Self {
        Self(value, T::default(), T::default())
    }
}

/// `polynomial!` allows `Polynomial`s to be defined with an arbitrary number of values.
///
/// Note that the arguments are used to value-initialize the underlying fixed-size tuple, and may
/// not exceed the container capacity.
#[macro_export]
macro_rules! polynomial {
    () => {
        $crate::ops::polynomial::Polynomial::default()
    };
    ($c0:expr) => {
        $crate::ops::polynomial::Polynomial::from($c0)
    };
    ($c0:expr, $c1:expr) => {
        $crate::ops::polynomial::Polynomial::new($c0, $c1, ::core::default::Default::default())
    };
    ($c0:expr, $c1:expr, $c2:expr) => {
        $crate::ops::polynomial::Polynomial::new($c0, $c1, $c2)
    };
}

macro_rules! offset_op_impl_scalar {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl<T> $trait<T> for Polynomial<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                Self(self.0.$method(rhs), self.1, self.2)
            }
        }

        impl<T> $assign_trait<T> for Polynomial<T>
        where
            T: $assign_trait + Copy,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: T) {
                self.0.$assign_method(rhs);
            }
        }
    };
}

offset_op_impl_scalar!(Add, add, AddAssign, add_assign);
offset_op_impl_scalar!(Sub, sub, SubAssign, sub_assign);

macro_rules! scale_op_impl_scalar {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl<T> $trait<T> for Polynomial<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                Self(self.0.$method(rhs), self.1.$method(rhs), self.2.$method(rhs))
            }
        }

        impl<T> $assign_trait<T> for Polynomial<T>
        where
            T: $assign_trait + Copy,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: T) {
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

macro_rules! offset_op_impl_polynomial {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl<T> $trait for Polynomial<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                Self(self.0.$method(rhs.0), self.1.$method(rhs.1), self.2.$method(rhs.2))
            }
        }

        impl<T> $assign_trait for Polynomial<T>
        where
            T: $assign_trait + Copy,
        {
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

impl<T> Mul for Polynomial<T>
where
    T: Add<Output = T> + Copy + Mul<Output = T>,
{
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

impl<T> MulAssign for Polynomial<T>
where
    T: Add<Output = T> + Copy + Mul<Output = T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self(
            self.0 * rhs.0,
            self.0 * rhs.1 + self.1 * rhs.0,
            self.0 * rhs.2 + self.1 * rhs.1 + self.2 * rhs.0,
        )
    }
}

impl<T> Neg for Polynomial<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1, -self.2)
    }
}
