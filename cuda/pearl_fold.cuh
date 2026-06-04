// ariaminer v0.4.0 — primitives de fold jackpot PORTÉES de la source officielle Pearl
// (pearl-gemm/csrc/gemm/pow_utils.hpp, ISC). Transcription 1:1 (pas deviné).
//   rotl_xor<13>  : mixer rotate-left(13) ^ y  (HASH_ACCUMULATE_ROTATION=13)
//   xor3_lop3     : XOR 3 entrées en 1 instruction lop3 (LUT 0x96)
//   xor_reduction : réduit en arbre lop3 TOUT le fragment accumulateur d'un thread -> 1 u32
// Ces 3 primitives sont arch-indépendantes (PTX standard) → tournent sur sm_120 sans Hopper.
#pragma once
#include <cstdint>
#include <cute/tensor.hpp>

namespace pearl_fold {

static constexpr int HASH_ACCUMULATE_ROTATION = 13;
static constexpr int JACKPOT_SIZE = 16;   // = blake3 MSG_BLOCK_SIZE_U32

// XOR 3 entrées via lop3 (LUT 0x96 = a^b^c)
__device__ __forceinline__ uint32_t xor3_lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// rotl(x,shift) ^ y
template <int shift>
__device__ __forceinline__ uint32_t rotl_xor(uint32_t x, uint32_t y) {
  static_assert(shift > 0 && shift < 32, "shift in (0,32)");
  uint32_t r;
  asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(r) : "r"(x), "n"(shift));
  return r ^ y;
}

// XOR-réduction (arbre lop3) de tous les éléments d'un fragment CuTe (registres).
template <typename TensorType>
__device__ __forceinline__ uint32_t xor_reduction(const TensorType& t) {
  constexpr int N = decltype(size(t))::value;
  static_assert(N > 0, "");
  uint32_t acc = (uint32_t)t(0);
  // arbre lop3 par triplets (équivalent au cute::fold officiel, déroulé linéaire)
  int i = 1;
  for (; i + 2 < N; i += 2) acc = xor3_lop3(acc, (uint32_t)t(i), (uint32_t)t(i+1));
  for (; i < N; ++i) acc ^= (uint32_t)t(i);
  return acc;
}

}  // namespace pearl_fold
