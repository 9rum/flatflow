// SPDX-License-Identifier: Apache-2.0

#include "flatflow/ops/scalar_type.h"

#include "flatflow/ops/scalar_type_generated.h"

using flatflow::promote_types;
using flatflow::ScalarType;
using flatflow::to_scale;

// Tests whether `promote_types` is idempotent.
static_assert(promote_types(ScalarType::float32, ScalarType::float32) ==
              ScalarType::float32);
static_assert(promote_types(ScalarType::float64, ScalarType::float64) ==
              ScalarType::float64);
static_assert(promote_types(ScalarType::float16, ScalarType::float16) ==
              ScalarType::float16);
static_assert(promote_types(ScalarType::bfloat16, ScalarType::bfloat16) ==
              ScalarType::bfloat16);
static_assert(promote_types(ScalarType::bool_, ScalarType::bool_) ==
              ScalarType::bool_);
static_assert(promote_types(ScalarType::int8, ScalarType::int8) ==
              ScalarType::int8);
static_assert(promote_types(ScalarType::int16, ScalarType::int16) ==
              ScalarType::int16);
static_assert(promote_types(ScalarType::int32, ScalarType::int32) ==
              ScalarType::int32);
static_assert(promote_types(ScalarType::int64, ScalarType::int64) ==
              ScalarType::int64);
static_assert(promote_types(ScalarType::uint8, ScalarType::uint8) ==
              ScalarType::uint8);
static_assert(promote_types(ScalarType::uint16, ScalarType::uint16) ==
              ScalarType::uint16);
static_assert(promote_types(ScalarType::uint32, ScalarType::uint32) ==
              ScalarType::uint32);
static_assert(promote_types(ScalarType::uint64, ScalarType::uint64) ==
              ScalarType::uint64);
static_assert(promote_types(ScalarType::float8_e4m3fn,
                            ScalarType::float8_e4m3fn) ==
              ScalarType::float8_e4m3fn);
static_assert(promote_types(ScalarType::float8_e4m3fnuz,
                            ScalarType::float8_e4m3fnuz) ==
              ScalarType::float8_e4m3fnuz);
static_assert(promote_types(ScalarType::float8_e5m2, ScalarType::float8_e5m2) ==
              ScalarType::float8_e5m2);
static_assert(promote_types(ScalarType::float8_e5m2fnuz,
                            ScalarType::float8_e5m2fnuz) ==
              ScalarType::float8_e5m2fnuz);

// Tests whether `promote_types` is commutative.
static_assert(promote_types(ScalarType::float32, ScalarType::float64) ==
              promote_types(ScalarType::float64, ScalarType::float32));
static_assert(promote_types(ScalarType::float32, ScalarType::float16) ==
              promote_types(ScalarType::float16, ScalarType::float32));
static_assert(promote_types(ScalarType::float32, ScalarType::bfloat16) ==
              promote_types(ScalarType::bfloat16, ScalarType::float32));
static_assert(promote_types(ScalarType::float32, ScalarType::bool_) ==
              promote_types(ScalarType::bool_, ScalarType::float32));
static_assert(promote_types(ScalarType::float32, ScalarType::int8) ==
              promote_types(ScalarType::int8, ScalarType::float32));
static_assert(promote_types(ScalarType::float32, ScalarType::int16) ==
              promote_types(ScalarType::int16, ScalarType::float32));
static_assert(promote_types(ScalarType::float32, ScalarType::int32) ==
              promote_types(ScalarType::int32, ScalarType::float32));
static_assert(promote_types(ScalarType::float32, ScalarType::int64) ==
              promote_types(ScalarType::int64, ScalarType::float32));
static_assert(promote_types(ScalarType::float32, ScalarType::uint8) ==
              promote_types(ScalarType::uint8, ScalarType::float32));

static_assert(promote_types(ScalarType::float64, ScalarType::float16) ==
              promote_types(ScalarType::float16, ScalarType::float64));
static_assert(promote_types(ScalarType::float64, ScalarType::bfloat16) ==
              promote_types(ScalarType::bfloat16, ScalarType::float64));
static_assert(promote_types(ScalarType::float64, ScalarType::bool_) ==
              promote_types(ScalarType::bool_, ScalarType::float64));
static_assert(promote_types(ScalarType::float64, ScalarType::int8) ==
              promote_types(ScalarType::int8, ScalarType::float64));
static_assert(promote_types(ScalarType::float64, ScalarType::int16) ==
              promote_types(ScalarType::int16, ScalarType::float64));
static_assert(promote_types(ScalarType::float64, ScalarType::int32) ==
              promote_types(ScalarType::int32, ScalarType::float64));
static_assert(promote_types(ScalarType::float64, ScalarType::int64) ==
              promote_types(ScalarType::int64, ScalarType::float64));
static_assert(promote_types(ScalarType::float64, ScalarType::uint8) ==
              promote_types(ScalarType::uint8, ScalarType::float64));

static_assert(promote_types(ScalarType::float16, ScalarType::bfloat16) ==
              promote_types(ScalarType::bfloat16, ScalarType::float16));
static_assert(promote_types(ScalarType::float16, ScalarType::bool_) ==
              promote_types(ScalarType::bool_, ScalarType::float16));
static_assert(promote_types(ScalarType::float16, ScalarType::int8) ==
              promote_types(ScalarType::int8, ScalarType::float16));
static_assert(promote_types(ScalarType::float16, ScalarType::int16) ==
              promote_types(ScalarType::int16, ScalarType::float16));
static_assert(promote_types(ScalarType::float16, ScalarType::int32) ==
              promote_types(ScalarType::int32, ScalarType::float16));
static_assert(promote_types(ScalarType::float16, ScalarType::int64) ==
              promote_types(ScalarType::int64, ScalarType::float16));
static_assert(promote_types(ScalarType::float16, ScalarType::uint8) ==
              promote_types(ScalarType::uint8, ScalarType::float16));

static_assert(promote_types(ScalarType::bfloat16, ScalarType::bool_) ==
              promote_types(ScalarType::bool_, ScalarType::bfloat16));
static_assert(promote_types(ScalarType::bfloat16, ScalarType::int8) ==
              promote_types(ScalarType::int8, ScalarType::bfloat16));
static_assert(promote_types(ScalarType::bfloat16, ScalarType::int16) ==
              promote_types(ScalarType::int16, ScalarType::bfloat16));
static_assert(promote_types(ScalarType::bfloat16, ScalarType::int32) ==
              promote_types(ScalarType::int32, ScalarType::bfloat16));
static_assert(promote_types(ScalarType::bfloat16, ScalarType::int64) ==
              promote_types(ScalarType::int64, ScalarType::bfloat16));
static_assert(promote_types(ScalarType::bfloat16, ScalarType::uint8) ==
              promote_types(ScalarType::uint8, ScalarType::bfloat16));

static_assert(promote_types(ScalarType::bool_, ScalarType::int8) ==
              promote_types(ScalarType::int8, ScalarType::bool_));
static_assert(promote_types(ScalarType::bool_, ScalarType::int16) ==
              promote_types(ScalarType::int16, ScalarType::bool_));
static_assert(promote_types(ScalarType::bool_, ScalarType::int32) ==
              promote_types(ScalarType::int32, ScalarType::bool_));
static_assert(promote_types(ScalarType::bool_, ScalarType::int64) ==
              promote_types(ScalarType::int64, ScalarType::bool_));
static_assert(promote_types(ScalarType::bool_, ScalarType::uint8) ==
              promote_types(ScalarType::uint8, ScalarType::bool_));

static_assert(promote_types(ScalarType::int8, ScalarType::int16) ==
              promote_types(ScalarType::int16, ScalarType::int8));
static_assert(promote_types(ScalarType::int8, ScalarType::int32) ==
              promote_types(ScalarType::int32, ScalarType::int8));
static_assert(promote_types(ScalarType::int8, ScalarType::int64) ==
              promote_types(ScalarType::int64, ScalarType::int8));
static_assert(promote_types(ScalarType::int8, ScalarType::uint8) ==
              promote_types(ScalarType::uint8, ScalarType::int8));

static_assert(promote_types(ScalarType::int16, ScalarType::int32) ==
              promote_types(ScalarType::int32, ScalarType::int16));
static_assert(promote_types(ScalarType::int16, ScalarType::int64) ==
              promote_types(ScalarType::int64, ScalarType::int16));
static_assert(promote_types(ScalarType::int16, ScalarType::uint8) ==
              promote_types(ScalarType::uint8, ScalarType::int16));

static_assert(promote_types(ScalarType::int32, ScalarType::int64) ==
              promote_types(ScalarType::int64, ScalarType::int32));
static_assert(promote_types(ScalarType::int32, ScalarType::uint8) ==
              promote_types(ScalarType::uint8, ScalarType::int32));

static_assert(promote_types(ScalarType::int64, ScalarType::uint8) ==
              promote_types(ScalarType::uint8, ScalarType::int64));

// Tests whether `promote_types` matches the C10 reference.
static_assert(promote_types(ScalarType::float32, ScalarType::float64) ==
              ScalarType::float64);
static_assert(promote_types(ScalarType::float32, ScalarType::float16) ==
              ScalarType::float32);
static_assert(promote_types(ScalarType::float32, ScalarType::bfloat16) ==
              ScalarType::float32);
static_assert(promote_types(ScalarType::float32, ScalarType::bool_) ==
              ScalarType::float32);
static_assert(promote_types(ScalarType::float32, ScalarType::int8) ==
              ScalarType::float32);
static_assert(promote_types(ScalarType::float32, ScalarType::int16) ==
              ScalarType::float32);
static_assert(promote_types(ScalarType::float32, ScalarType::int32) ==
              ScalarType::float32);
static_assert(promote_types(ScalarType::float32, ScalarType::int64) ==
              ScalarType::float32);
static_assert(promote_types(ScalarType::float32, ScalarType::uint8) ==
              ScalarType::float32);

static_assert(promote_types(ScalarType::float64, ScalarType::float16) ==
              ScalarType::float64);
static_assert(promote_types(ScalarType::float64, ScalarType::bfloat16) ==
              ScalarType::float64);
static_assert(promote_types(ScalarType::float64, ScalarType::bool_) ==
              ScalarType::float64);
static_assert(promote_types(ScalarType::float64, ScalarType::int8) ==
              ScalarType::float64);
static_assert(promote_types(ScalarType::float64, ScalarType::int16) ==
              ScalarType::float64);
static_assert(promote_types(ScalarType::float64, ScalarType::int32) ==
              ScalarType::float64);
static_assert(promote_types(ScalarType::float64, ScalarType::int64) ==
              ScalarType::float64);
static_assert(promote_types(ScalarType::float64, ScalarType::uint8) ==
              ScalarType::float64);

static_assert(promote_types(ScalarType::float16, ScalarType::bfloat16) ==
              ScalarType::float32);
static_assert(promote_types(ScalarType::float16, ScalarType::bool_) ==
              ScalarType::float16);
static_assert(promote_types(ScalarType::float16, ScalarType::int8) ==
              ScalarType::float16);
static_assert(promote_types(ScalarType::float16, ScalarType::int16) ==
              ScalarType::float16);
static_assert(promote_types(ScalarType::float16, ScalarType::int32) ==
              ScalarType::float16);
static_assert(promote_types(ScalarType::float16, ScalarType::int64) ==
              ScalarType::float16);
static_assert(promote_types(ScalarType::float16, ScalarType::uint8) ==
              ScalarType::float16);

static_assert(promote_types(ScalarType::bfloat16, ScalarType::bool_) ==
              ScalarType::bfloat16);
static_assert(promote_types(ScalarType::bfloat16, ScalarType::int8) ==
              ScalarType::bfloat16);
static_assert(promote_types(ScalarType::bfloat16, ScalarType::int16) ==
              ScalarType::bfloat16);
static_assert(promote_types(ScalarType::bfloat16, ScalarType::int32) ==
              ScalarType::bfloat16);
static_assert(promote_types(ScalarType::bfloat16, ScalarType::int64) ==
              ScalarType::bfloat16);
static_assert(promote_types(ScalarType::bfloat16, ScalarType::uint8) ==
              ScalarType::bfloat16);

static_assert(promote_types(ScalarType::bool_, ScalarType::int8) ==
              ScalarType::int8);
static_assert(promote_types(ScalarType::bool_, ScalarType::int16) ==
              ScalarType::int16);
static_assert(promote_types(ScalarType::bool_, ScalarType::int32) ==
              ScalarType::int32);
static_assert(promote_types(ScalarType::bool_, ScalarType::int64) ==
              ScalarType::int64);
static_assert(promote_types(ScalarType::bool_, ScalarType::uint8) ==
              ScalarType::uint8);

static_assert(promote_types(ScalarType::int8, ScalarType::int16) ==
              ScalarType::int16);
static_assert(promote_types(ScalarType::int8, ScalarType::int32) ==
              ScalarType::int32);
static_assert(promote_types(ScalarType::int8, ScalarType::int64) ==
              ScalarType::int64);
static_assert(promote_types(ScalarType::int8, ScalarType::uint8) ==
              ScalarType::int16);

static_assert(promote_types(ScalarType::int16, ScalarType::int32) ==
              ScalarType::int32);
static_assert(promote_types(ScalarType::int16, ScalarType::int64) ==
              ScalarType::int64);
static_assert(promote_types(ScalarType::int16, ScalarType::uint8) ==
              ScalarType::int16);

static_assert(promote_types(ScalarType::int32, ScalarType::int64) ==
              ScalarType::int64);
static_assert(promote_types(ScalarType::int32, ScalarType::uint8) ==
              ScalarType::int32);

static_assert(promote_types(ScalarType::int64, ScalarType::uint8) ==
              ScalarType::int64);

// Tests whether `to_scale` returns the correct scale factor.
static_assert(to_scale(ScalarType::float32) == 4);
static_assert(to_scale(ScalarType::float64) == 64);
static_assert(to_scale(ScalarType::float16) == 2);
static_assert(to_scale(ScalarType::bfloat16) == 2);
static_assert(to_scale(ScalarType::bool_) == 64);
static_assert(to_scale(ScalarType::int8) == 1);
static_assert(to_scale(ScalarType::int16) == 64);
static_assert(to_scale(ScalarType::int32) == 64);
static_assert(to_scale(ScalarType::int64) == 128);
static_assert(to_scale(ScalarType::uint8) == 1);
static_assert(to_scale(ScalarType::uint16) == 64);
static_assert(to_scale(ScalarType::uint32) == 64);
static_assert(to_scale(ScalarType::uint64) == 128);
static_assert(to_scale(ScalarType::float8_e4m3fn) == 1);
static_assert(to_scale(ScalarType::float8_e4m3fnuz) == 1);
static_assert(to_scale(ScalarType::float8_e5m2) == 1);
static_assert(to_scale(ScalarType::float8_e5m2fnuz) == 1);
