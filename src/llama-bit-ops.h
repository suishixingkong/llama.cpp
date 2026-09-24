#pragma once

#include <cstdint>

// C++17-compatible bit operations used by the VBR metadata paths. Keep these
// independent of compiler-specific builtins so the same sources compile with
// MSVC, GCC, and Clang. Optimizing compilers recognize these fixed-width idioms
// and lower them to native bit operations when available.
constexpr uint32_t llama_popcount_u64(uint64_t value) noexcept {
    value -= (value >> 1) & UINT64_C(0x5555555555555555);
    value  = (value & UINT64_C(0x3333333333333333)) +
             ((value >> 2) & UINT64_C(0x3333333333333333));
    value  = (value + (value >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return uint32_t((value * UINT64_C(0x0101010101010101)) >> 56);
}

// Match the conventional countr_zero result for zero. The ownership iterator
// normally passes nonzero words, but defining zero keeps the helper total.
constexpr uint32_t llama_countr_zero_u64(uint64_t value) noexcept {
    return llama_popcount_u64((value & (UINT64_C(0) - value)) - UINT64_C(1));
}

static_assert(llama_popcount_u64(UINT64_C(0)) == 0, "zero popcount");
static_assert(llama_popcount_u64(UINT64_MAX) == 64, "full popcount");
static_assert(llama_countr_zero_u64(UINT64_C(1)) == 0, "low-bit ctz");
static_assert(llama_countr_zero_u64(UINT64_C(1) << 63) == 63, "high-bit ctz");
static_assert(llama_countr_zero_u64(UINT64_C(0)) == 64, "zero ctz");
