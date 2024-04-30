// Copyright 2024 The FlatFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FLATFLOW_BASE_ATTRIBUTES_H_
#define FLATFLOW_BASE_ATTRIBUTES_H_

// FLATFLOW_ATTRIBUTE_ALWAYS_INLINE
// FLATFLOW_ATTRIBUTE_NOINLINE
//
// Forces functions to either inline or not inline. Introduced in GCC 3.1.
#ifdef __GNUC__
#define FLATFLOW_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#ifdef _MSC_VER
#define FLATFLOW_ATTRIBUTE_ALWAYS_INLINE __forceinline
#else
#define FLATFLOW_ATTRIBUTE_ALWAYS_INLINE inline
#endif  // _MSC_VER
#endif  // __GNUC__

#ifdef __GNUC__
#define FLATFLOW_ATTRIBUTE_NOINLINE __attribute__((noinline))
#else
#ifdef _MSC_VER
#define FLATFLOW_ATTRIBUTE_NOINLINE __declspec(noinline)
#else
#define FLATFLOW_ATTRIBUTE_NOINLINE
#endif  // _MSC_VER
#endif  // __GNUC__

// FLATFLOW_ATTRIBUTE_RESTRICT
//
// Indicates that the symbol is not aliased in the local context.
#ifdef __GNUC__
#define FLATFLOW_ATTRIBUTE_RESTRICT __restrict__
#else
#ifdef _MSC_VER
#define FLATFLOW_ATTRIBUTE_RESTRICT __restrict
#else
#define FLATFLOW_ATTRIBUTE_RESTRICT
#endif  // _MSC_VER
#endif  // __GNUC__

#endif  // FLATFLOW_BASE_ATTRIBUTES_H_
