// SPDX-License-Identifier: Apache-2.0

//! Type promotion utility.

use crate::ops::scalar_type_generated::ScalarType;

impl ScalarType {
    #[inline]
    const fn is_float8_type(self) -> bool {
        matches!(
            self,
            Self::FLOAT8_E4M3FN | Self::FLOAT8_E4M3FNUZ | Self::FLOAT8_E5M2 | Self::FLOAT8_E5M2FNUZ
        )
    }

    #[inline]
    const fn is_reduced_floating_type(self) -> bool {
        matches!(self, Self::FLOAT16 | Self::BFLOAT16) || self.is_float8_type()
    }

    #[inline]
    const fn is_floating_type(self) -> bool {
        matches!(self, Self::FLOAT32 | Self::FLOAT64) || self.is_reduced_floating_type()
    }

    #[inline]
    const fn is_barebones_unsigned_type(self) -> bool {
        matches!(self, Self::UINT16 | Self::UINT32 | Self::UINT64)
    }
}

impl From<ScalarType> for i64 {
    /// Maps the given data type to the corresponding FLOPS scale.
    #[inline]
    fn from(dtype: ScalarType) -> Self {
        match dtype {
            ScalarType::INT8
            | ScalarType::UINT8
            | ScalarType::FLOAT8_E4M3FN
            | ScalarType::FLOAT8_E4M3FNUZ
            | ScalarType::FLOAT8_E5M2
            | ScalarType::FLOAT8_E5M2FNUZ => 1,
            ScalarType::FLOAT16 | ScalarType::BFLOAT16 => 2,
            ScalarType::FLOAT32 => 4,
            ScalarType::FLOAT64
            | ScalarType::BOOL
            | ScalarType::INT16
            | ScalarType::INT32
            | ScalarType::UINT16
            | ScalarType::UINT32 => 64,
            ScalarType::INT64 | ScalarType::UINT64 => 128,
            _ => unreachable!(),
        }
    }
}

/// Returns the data type with the smallest size and scalar kind that is not smaller nor of lower
/// kind than either `lhs` or `rhs`. See [type promotion documentation] for more information on the
/// type promotion logic.
///
/// [type promotion documentation]: https://docs.pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc
#[inline]
pub const fn promote_types(lhs: ScalarType, rhs: ScalarType) -> ScalarType {
    // If the two types are equal, return that type.
    if lhs.0 == rhs.0 {
        return lhs;
    }

    assert!(
        !(lhs.is_float8_type() || rhs.is_float8_type()),
        "Promotion for float8 types is not supported",
    );

    if lhs.is_barebones_unsigned_type() || rhs.is_barebones_unsigned_type() {
        if lhs.is_floating_type() {
            return lhs;
        }
        if rhs.is_floating_type() {
            return rhs;
        }
        panic!("Promotion for uint16, uint32, uint64 types is not supported");
    }

    const F4: ScalarType = ScalarType::FLOAT32;
    const F8: ScalarType = ScalarType::FLOAT64;
    const F2: ScalarType = ScalarType::FLOAT16;
    const BF: ScalarType = ScalarType::BFLOAT16;
    const B1: ScalarType = ScalarType::BOOL;
    const I1: ScalarType = ScalarType::INT8;
    const I2: ScalarType = ScalarType::INT16;
    const I4: ScalarType = ScalarType::INT32;
    const I8: ScalarType = ScalarType::INT64;
    const U1: ScalarType = ScalarType::UINT8;

    const LOOKUP: [[ScalarType; 10]; 10] = [
        /*        F4  F8  F2  BF  B1  I1  I2  I4  I8  U1 */
        [/* F4 */ F4, F8, F4, F4, F4, F4, F4, F4, F4, F4],
        [/* F8 */ F8, F8, F8, F8, F8, F8, F8, F8, F8, F8],
        [/* F2 */ F4, F8, F2, F4, F2, F2, F2, F2, F2, F2],
        [/* BF */ F4, F8, F4, BF, BF, BF, BF, BF, BF, BF],
        [/* B1 */ F4, F8, F2, BF, B1, I1, I2, I4, I8, U1],
        [/* I1 */ F4, F8, F2, BF, I1, I1, I2, I4, I8, I2],
        [/* I2 */ F4, F8, F2, BF, I2, I2, I2, I4, I8, I2],
        [/* I4 */ F4, F8, F2, BF, I4, I4, I4, I4, I8, I4],
        [/* I8 */ F4, F8, F2, BF, I8, I8, I8, I8, I8, I8],
        [/* U1 */ F4, F8, F2, BF, U1, I2, I2, I4, I8, U1],
    ];

    LOOKUP[lhs.0 as usize][rhs.0 as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_types_has_diagonal_identity() {
        for &dtype in ScalarType::ENUM_VALUES {
            assert_eq!(promote_types(dtype, dtype), dtype);
        }
    }

    #[test]
    fn test_promote_types_is_commutative() {
        assert_eq!(
            promote_types(ScalarType::FLOAT32, ScalarType::FLOAT64),
            promote_types(ScalarType::FLOAT64, ScalarType::FLOAT32),
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT32, ScalarType::FLOAT16),
            promote_types(ScalarType::FLOAT16, ScalarType::FLOAT32)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT32, ScalarType::BFLOAT16),
            promote_types(ScalarType::BFLOAT16, ScalarType::FLOAT32)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT32, ScalarType::BOOL),
            promote_types(ScalarType::BOOL, ScalarType::FLOAT32)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT32, ScalarType::INT8),
            promote_types(ScalarType::INT8, ScalarType::FLOAT32)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT32, ScalarType::INT16),
            promote_types(ScalarType::INT16, ScalarType::FLOAT32)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT32, ScalarType::INT32),
            promote_types(ScalarType::INT32, ScalarType::FLOAT32)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT32, ScalarType::INT64),
            promote_types(ScalarType::INT64, ScalarType::FLOAT32)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT32, ScalarType::UINT8),
            promote_types(ScalarType::UINT8, ScalarType::FLOAT32)
        );

        assert_eq!(
            promote_types(ScalarType::FLOAT64, ScalarType::FLOAT16),
            promote_types(ScalarType::FLOAT16, ScalarType::FLOAT64)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT64, ScalarType::BFLOAT16),
            promote_types(ScalarType::BFLOAT16, ScalarType::FLOAT64)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT64, ScalarType::BOOL),
            promote_types(ScalarType::BOOL, ScalarType::FLOAT64)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT64, ScalarType::INT8),
            promote_types(ScalarType::INT8, ScalarType::FLOAT64)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT64, ScalarType::INT16),
            promote_types(ScalarType::INT16, ScalarType::FLOAT64)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT64, ScalarType::INT32),
            promote_types(ScalarType::INT32, ScalarType::FLOAT64)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT64, ScalarType::INT64),
            promote_types(ScalarType::INT64, ScalarType::FLOAT64)
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT64, ScalarType::UINT8),
            promote_types(ScalarType::UINT8, ScalarType::FLOAT64)
        );

        assert_eq!(
            promote_types(ScalarType::FLOAT16, ScalarType::BFLOAT16),
            promote_types(ScalarType::BFLOAT16, ScalarType::FLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT16, ScalarType::BOOL),
            promote_types(ScalarType::BOOL, ScalarType::FLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT16, ScalarType::INT8),
            promote_types(ScalarType::INT8, ScalarType::FLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT16, ScalarType::INT16),
            promote_types(ScalarType::INT16, ScalarType::FLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT16, ScalarType::INT32),
            promote_types(ScalarType::INT32, ScalarType::FLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT16, ScalarType::INT64),
            promote_types(ScalarType::INT64, ScalarType::FLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::FLOAT16, ScalarType::UINT8),
            promote_types(ScalarType::UINT8, ScalarType::FLOAT16),
        );

        assert_eq!(
            promote_types(ScalarType::BFLOAT16, ScalarType::BOOL),
            promote_types(ScalarType::BOOL, ScalarType::BFLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::BFLOAT16, ScalarType::INT8),
            promote_types(ScalarType::INT8, ScalarType::BFLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::BFLOAT16, ScalarType::INT16),
            promote_types(ScalarType::INT16, ScalarType::BFLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::BFLOAT16, ScalarType::INT32),
            promote_types(ScalarType::INT32, ScalarType::BFLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::BFLOAT16, ScalarType::INT64),
            promote_types(ScalarType::INT64, ScalarType::BFLOAT16),
        );
        assert_eq!(
            promote_types(ScalarType::BFLOAT16, ScalarType::UINT8),
            promote_types(ScalarType::UINT8, ScalarType::BFLOAT16),
        );

        assert_eq!(
            promote_types(ScalarType::BOOL, ScalarType::INT8),
            promote_types(ScalarType::INT8, ScalarType::BOOL),
        );
        assert_eq!(
            promote_types(ScalarType::BOOL, ScalarType::INT16),
            promote_types(ScalarType::INT16, ScalarType::BOOL),
        );
        assert_eq!(
            promote_types(ScalarType::BOOL, ScalarType::INT32),
            promote_types(ScalarType::INT32, ScalarType::BOOL),
        );
        assert_eq!(
            promote_types(ScalarType::BOOL, ScalarType::INT64),
            promote_types(ScalarType::INT64, ScalarType::BOOL),
        );
        assert_eq!(
            promote_types(ScalarType::BOOL, ScalarType::UINT8),
            promote_types(ScalarType::UINT8, ScalarType::BOOL),
        );

        assert_eq!(
            promote_types(ScalarType::INT8, ScalarType::INT16),
            promote_types(ScalarType::INT16, ScalarType::INT8),
        );
        assert_eq!(
            promote_types(ScalarType::INT8, ScalarType::INT32),
            promote_types(ScalarType::INT32, ScalarType::INT8),
        );
        assert_eq!(
            promote_types(ScalarType::INT8, ScalarType::INT64),
            promote_types(ScalarType::INT64, ScalarType::INT8),
        );
        assert_eq!(
            promote_types(ScalarType::INT8, ScalarType::UINT8),
            promote_types(ScalarType::UINT8, ScalarType::INT8),
        );

        assert_eq!(
            promote_types(ScalarType::INT16, ScalarType::INT32),
            promote_types(ScalarType::INT32, ScalarType::INT16),
        );
        assert_eq!(
            promote_types(ScalarType::INT16, ScalarType::INT64),
            promote_types(ScalarType::INT64, ScalarType::INT16),
        );
        assert_eq!(
            promote_types(ScalarType::INT16, ScalarType::UINT8),
            promote_types(ScalarType::UINT8, ScalarType::INT16),
        );

        assert_eq!(
            promote_types(ScalarType::INT32, ScalarType::INT64),
            promote_types(ScalarType::INT64, ScalarType::INT32)
        );
        assert_eq!(
            promote_types(ScalarType::INT32, ScalarType::UINT8),
            promote_types(ScalarType::UINT8, ScalarType::INT32)
        );

        assert_eq!(
            promote_types(ScalarType::INT64, ScalarType::UINT8),
            promote_types(ScalarType::UINT8, ScalarType::INT64),
        );
    }

    #[test]
    fn test_promote_types_matches_c10_reference() {
        assert_eq!(promote_types(ScalarType::FLOAT32, ScalarType::FLOAT64), ScalarType::FLOAT64);
        assert_eq!(promote_types(ScalarType::FLOAT32, ScalarType::FLOAT16), ScalarType::FLOAT32);
        assert_eq!(promote_types(ScalarType::FLOAT32, ScalarType::BFLOAT16), ScalarType::FLOAT32);
        assert_eq!(promote_types(ScalarType::FLOAT32, ScalarType::BOOL), ScalarType::FLOAT32);
        assert_eq!(promote_types(ScalarType::FLOAT32, ScalarType::INT8), ScalarType::FLOAT32);
        assert_eq!(promote_types(ScalarType::FLOAT32, ScalarType::INT16), ScalarType::FLOAT32);
        assert_eq!(promote_types(ScalarType::FLOAT32, ScalarType::INT32), ScalarType::FLOAT32);
        assert_eq!(promote_types(ScalarType::FLOAT32, ScalarType::INT64), ScalarType::FLOAT32);
        assert_eq!(promote_types(ScalarType::FLOAT32, ScalarType::UINT8), ScalarType::FLOAT32);

        assert_eq!(promote_types(ScalarType::FLOAT64, ScalarType::FLOAT16), ScalarType::FLOAT64);
        assert_eq!(promote_types(ScalarType::FLOAT64, ScalarType::BFLOAT16), ScalarType::FLOAT64);
        assert_eq!(promote_types(ScalarType::FLOAT64, ScalarType::BOOL), ScalarType::FLOAT64);
        assert_eq!(promote_types(ScalarType::FLOAT64, ScalarType::INT8), ScalarType::FLOAT64);
        assert_eq!(promote_types(ScalarType::FLOAT64, ScalarType::INT16), ScalarType::FLOAT64);
        assert_eq!(promote_types(ScalarType::FLOAT64, ScalarType::INT32), ScalarType::FLOAT64);
        assert_eq!(promote_types(ScalarType::FLOAT64, ScalarType::INT64), ScalarType::FLOAT64);
        assert_eq!(promote_types(ScalarType::FLOAT64, ScalarType::UINT8), ScalarType::FLOAT64);

        assert_eq!(promote_types(ScalarType::FLOAT16, ScalarType::BFLOAT16), ScalarType::FLOAT32);
        assert_eq!(promote_types(ScalarType::FLOAT16, ScalarType::BOOL), ScalarType::FLOAT16);
        assert_eq!(promote_types(ScalarType::FLOAT16, ScalarType::INT8), ScalarType::FLOAT16);
        assert_eq!(promote_types(ScalarType::FLOAT16, ScalarType::INT16), ScalarType::FLOAT16);
        assert_eq!(promote_types(ScalarType::FLOAT16, ScalarType::INT32), ScalarType::FLOAT16);
        assert_eq!(promote_types(ScalarType::FLOAT16, ScalarType::INT64), ScalarType::FLOAT16);
        assert_eq!(promote_types(ScalarType::FLOAT16, ScalarType::UINT8), ScalarType::FLOAT16);

        assert_eq!(promote_types(ScalarType::BFLOAT16, ScalarType::BOOL), ScalarType::BFLOAT16);
        assert_eq!(promote_types(ScalarType::BFLOAT16, ScalarType::INT8), ScalarType::BFLOAT16);
        assert_eq!(promote_types(ScalarType::BFLOAT16, ScalarType::INT16), ScalarType::BFLOAT16);
        assert_eq!(promote_types(ScalarType::BFLOAT16, ScalarType::INT32), ScalarType::BFLOAT16);
        assert_eq!(promote_types(ScalarType::BFLOAT16, ScalarType::INT64), ScalarType::BFLOAT16);
        assert_eq!(promote_types(ScalarType::BFLOAT16, ScalarType::UINT8), ScalarType::BFLOAT16);

        assert_eq!(promote_types(ScalarType::BOOL, ScalarType::INT8), ScalarType::INT8);
        assert_eq!(promote_types(ScalarType::BOOL, ScalarType::INT16), ScalarType::INT16);
        assert_eq!(promote_types(ScalarType::BOOL, ScalarType::INT32), ScalarType::INT32);
        assert_eq!(promote_types(ScalarType::BOOL, ScalarType::INT64), ScalarType::INT64);
        assert_eq!(promote_types(ScalarType::BOOL, ScalarType::UINT8), ScalarType::UINT8);

        assert_eq!(promote_types(ScalarType::INT8, ScalarType::INT16), ScalarType::INT16);
        assert_eq!(promote_types(ScalarType::INT8, ScalarType::INT32), ScalarType::INT32);
        assert_eq!(promote_types(ScalarType::INT8, ScalarType::INT64), ScalarType::INT64);
        assert_eq!(promote_types(ScalarType::INT8, ScalarType::UINT8), ScalarType::INT16);

        assert_eq!(promote_types(ScalarType::INT16, ScalarType::INT32), ScalarType::INT32);
        assert_eq!(promote_types(ScalarType::INT16, ScalarType::INT64), ScalarType::INT64);
        assert_eq!(promote_types(ScalarType::INT16, ScalarType::UINT8), ScalarType::INT16);

        assert_eq!(promote_types(ScalarType::INT32, ScalarType::INT64), ScalarType::INT64);
        assert_eq!(promote_types(ScalarType::INT32, ScalarType::UINT8), ScalarType::INT32);

        assert_eq!(promote_types(ScalarType::INT64, ScalarType::UINT8), ScalarType::INT64);
    }

    #[test]
    #[should_panic]
    fn test_into_panics_on_undefined_scalar_type() {
        let _: i64 = ScalarType(u8::MAX).into();
    }
}
