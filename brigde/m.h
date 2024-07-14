
#include <torch/extension.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <torch/torch.h>
#include <ATen/cuda/PhiloxCudaState.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>

using index_t      = int32_t;
using long_index_t = int64_t;
using Index = index_t;

struct ihipStream_t;
typedef struct ihipStream_t* hipStream_t; 

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                     \
  [&] {                                                                        \
    if (COND) {                                                                \
      constexpr static bool CONST_NAME = true;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool CONST_NAME = false;                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define BF16_SWITCH(COND, ...)                                                 \
  [&] {                                                                        \
    if (COND) {                                                                \
      using T = device_gemm_trait::BFloat16;                                   \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      using T = device_gemm_trait::Float16;                                    \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)                                           \
  [&] {                                                                        \
    if (HEADDIM <= 32) {                                                       \
      constexpr static int kHeadDim = 32;                                      \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM <= 64) {                                                \
      constexpr static int kHeadDim = 64;                                      \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM <= 128) {                                               \
      constexpr static int kHeadDim = 128;                                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()


namespace device_gemm_trait{
    struct GemmSpecialization;
    struct GemmSpec;
}

