// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.


#include <torch/extension.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <torch/torch.h>
#include <ATen/cuda/PhiloxCudaState.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>

#include "m.h"

#define CHECK_SHAPE(x, ...)                                                    \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}),                  \
              #x " must have shape (" #__VA_ARGS__ ")")
              

#define NEW_UNPACK 1
static std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) {
  if (arg.captured_) {
#if NEW_UNPACK
    return std::make_tuple(
        static_cast<uint64_t>(*arg.seed_.ptr),
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
#else
    return std::make_tuple(
        arg.seed_,
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
#endif
  } else {
#if NEW_UNPACK
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
#else
    return std::make_tuple(arg.seed_, arg.offset_.val);
#endif
  }
}

// get environment variables for internal usage
static inline bool get_env_(const char *env_var) {
  if (char *value = std::getenv(env_var)) {
    if (strcmp(value, "0") == 0) {
      return false;
    }
    return true;
  }
  return false;
}

__declspec(dllimport)  struct BaseParams {
  explicit BaseParams(const Index b, const Index max_seqlen_q,
                      const Index max_seqlen_kv, const Index h_q,
                      const Index h_kv, const Index d, const torch::Tensor &q,
                      const torch::Tensor &k, const torch::Tensor &v,
                      torch::Tensor &out, torch::Tensor &softmax_lse,
                      const float p_dropout, const float softmax_scale,
                      const bool is_causal)
      : b(b), max_seqlen_q(max_seqlen_q), max_seqlen_kv(max_seqlen_kv),
        h_q(h_q), h_kv(h_kv), d(d), p_dropout(p_dropout),
        softmax_scale(softmax_scale), is_bf16(q.dtype() == torch::kBFloat16),
        is_dropout(p_dropout > 0.0f), is_mnko_padding(false),
        is_causal(is_causal), q_seq_stride(q.stride(-3)),
        kv_seq_stride(k.stride(-3)), out_seq_stride(out.stride(-3)),
        q_head_stride(q.stride(-2)), kv_head_stride(k.stride(-2)),
        out_head_stride(out.stride(-2)),
        softmax_lse_batch_stride(softmax_lse.stride(0)) {
    TORCH_CHECK(p_dropout < 1.f);
    is_mnko_padding = ((d % 32) != 0) || (d == 96);
    if (d > 128) {
      std::cout << "Unsupported head dimension" << std::endl;
    }
  }
  // The dimensions.
  Index b, max_seqlen_q, max_seqlen_kv, d;

  // The number of heads.
  Index h_q, h_kv;

  // The scaling factors for the kernel.
  float softmax_scale;
  // float softmax_scale_log2;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  // uint8_t p_dropout_in_uint8_t;

  // seeds
  std::tuple<uint64_t, uint64_t> seeds;

  bool is_bf16;
  bool is_dropout;
  bool is_mnko_padding;
  bool is_causal;

  Index q_seq_stride;
  Index kv_seq_stride;
  Index out_seq_stride;

  Index q_head_stride;
  Index kv_head_stride;
  Index out_head_stride;

  Index softmax_lse_batch_stride;

  static inline const bool kIsUnitTestMode =
      get_env_("FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE");
  static inline const bool kIsDeterministic =
      get_env_("FLASH_ATTENTION_INTERNAL_DETERMINISTIC");
};


__declspec(dllimport)  struct BatchedParams : public BaseParams {
  explicit BatchedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, const torch::Tensor &q,
      const torch::Tensor &k, const torch::Tensor &v, torch::Tensor &out,
      torch::Tensor
          &softmax_lse, // TODO: forward reference, backward const reference
      const float p_dropout, const float softmax_scale, const bool is_causal)
      : BaseParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, q, k, v, out,
                   softmax_lse, p_dropout, softmax_scale, is_causal),
        q_ptr(q.data_ptr()), k_ptr(k.data_ptr()), v_ptr(v.data_ptr()),
        out_ptr(out.data_ptr()), softmax_lse_ptr(softmax_lse.data_ptr()),
        q_batch_stride(q.stride(0)), kv_batch_stride(k.stride(0)),
        out_batch_stride(out.stride(0)) {
    if (!is_mnko_padding && d <= 32) {
      is_mnko_padding =
          ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                   : true);
    } else if (!is_mnko_padding && d <= 64) {
      if (is_dropout) {
        is_mnko_padding =
            ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                     : true);
      } else {
        is_mnko_padding =
            ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 256) == 0 ? false
                                                                     : true);
      }
    } else if (!is_mnko_padding && d <= 128) {
      is_mnko_padding =
          ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                   : true);
    }

    // TODO: Change to tensor.shape()
    // Q layout [b, max_seqlen_q, h_q, d]
    q_lengths = std::vector<Index>{b, h_q, max_seqlen_q, d};
    q_strides =
        std::vector<Index>{q_batch_stride, q_head_stride, q_seq_stride, 1};

    // K layout [b, max_seqlen_kv, h_kv, d]
    k_lengths = std::vector<Index>{b, h_kv, max_seqlen_kv, d};
    k_strides =
        std::vector<Index>{kv_batch_stride, kv_head_stride, kv_seq_stride, 1};

    // V layout [b, max_seqlen_kv, h_kv, d]
    v_lengths = std::vector<Index>{b, h_kv, d, max_seqlen_kv};
    v_strides =
        std::vector<Index>{kv_batch_stride, kv_head_stride, 1, kv_seq_stride};

    // Y layout [b, max_seqlen_q, h_q, d]
    out_lengths = std::vector<Index>{b, h_q, max_seqlen_q, d};
    out_strides = std::vector<Index>{out_batch_stride, out_head_stride,
                                     out_seq_stride, 1};

    // LSE layout [b, h_q, max_seqlen_q]
    lse_lengths = std::vector<Index>{b, h_q, max_seqlen_q};
    // std::vector<Index> lse_strides{h_q*max_seqlen_q, max_seqlen_q, 1};
  }

  void *__restrict__ q_ptr;
  void *__restrict__ k_ptr;
  void *__restrict__ v_ptr;
  void *__restrict__ z_ptr;
  void *__restrict__ out_ptr;
  void *__restrict__ softmax_lse_ptr;

  Index q_batch_stride;
  Index kv_batch_stride;
  Index out_batch_stride;
  Index softmax_lse_batch_stride;

  std::vector<Index> q_lengths;
  std::vector<Index> q_strides;
  std::vector<Index> k_lengths;
  std::vector<Index> k_strides;
  std::vector<Index> v_lengths;
  std::vector<Index> v_strides;
  std::vector<Index> z_lengths;
  std::vector<Index> z_strides;
  std::vector<Index> out_lengths;
  std::vector<Index> out_strides;
  std::vector<Index> lse_lengths;
  // std::vector<Index> lse_strides;
};

__declspec(dllimport)  // Forward Batched Arguments
struct FlashFwdBatchedParams : public BatchedParams {
  explicit FlashFwdBatchedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, const torch::Tensor &q,
      const torch::Tensor &k, const torch::Tensor &v, torch::Tensor &out,
      torch::Tensor &z,
      torch::Tensor
          &softmax_lse, // TODO: forward reference, backward const reference
      const float p_dropout, const float softmax_scale, const bool is_causal,
      const bool return_softmax)
      : BatchedParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, q, k, v,
                      out, softmax_lse, p_dropout, softmax_scale, is_causal) {
    z_ptr = return_softmax ? z.data_ptr() : nullptr;

    // Z layout [b, h_q, max_seqlen_q, max_seqlen_kv]
    z_lengths = std::vector<Index>{b, h_q, max_seqlen_q, max_seqlen_kv};
    z_strides =
        std::vector<Index>{h_q * max_seqlen_q * max_seqlen_kv,
                           max_seqlen_q * max_seqlen_kv, max_seqlen_kv, 1};
  }

  bool return_softmax;
};



void __declspec(dllimport)  flash_run_fwd_batch(FlashFwdBatchedParams bp, void *s);
//void __declspec(dllimport)  flash_run_fwd_group(FlashFwdGroupedParams bp, void *s);
//void __declspec(dllimport)  flash_run_bwd_batch(FlashBwdBatchedParams bp, void *s);
//void __declspec(dllimport)  flash_run_bwd_group(FlashBwdGroupedParams bp, void *s);


 __declspec(dllimport)  bool chk_is_wmma_supported() ;
 __declspec(dllimport)  bool chk_is_xdl_supported();
 
std::vector<torch::Tensor> mha_fwd(
    const torch::Tensor &q, // batch_size x seqlen_q x num_heads_q x head_size
    const torch::Tensor &k, // batch_size x seqlen_kv x num_heads_kv x head_size
    const torch::Tensor &v, // batch_size x seqlen_kv x num_heads_kv x head_size
    c10::optional<torch::Tensor>
        &out_, // batch_size x seqlen_q x num_heads_q x head_size
    const float p_dropout, const float softmax_scale, const bool is_causal,
    const bool return_softmax, c10::optional<at::Generator> gen_) {

  TORCH_CHECK(
      chk_is_xdl_supported() || chk_is_wmma_supported(),
      "FlashAttention currently only supports MI100 and RX7000 and above");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");

  TORCH_CHECK(k.dtype() == q_dtype, "Query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "Query and value must have the same dtype");

  TORCH_CHECK(q.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(k.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(v.is_cuda(), "Input tensor must be on ROCm device");

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads_q = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_kv = k.size(1);
  const int num_heads_kv = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be postive");
  TORCH_CHECK(
      head_size_og <= 128,
      "FlashAttention forward only supports head dimension at most 128");
  TORCH_CHECK(
      num_heads_q % num_heads_kv == 0,
      "Number of heads in key/value must divide number of heads in Query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
              "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads_q, head_size_og);
  CHECK_SHAPE(k, batch_size, seqlen_kv, num_heads_kv, head_size_og);
  CHECK_SHAPE(v, batch_size, seqlen_kv, num_heads_kv, head_size_og);

  torch::Tensor q_padded, k_padded, v_padded;
  if (head_size_og % 8 != 0) {
    q_padded = torch::nn::functional::pad(
        q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    k_padded = torch::nn::functional::pad(
        k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    v_padded = torch::nn::functional::pad(
        v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  } else {
    q_padded = q;
    k_padded = k;
    v_padded = v;
//#if defined(__WMMA__)
    q_padded = q.contiguous();
    k_padded = k.contiguous();
    v_padded = v.contiguous();
//#endif
  }

  torch::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype,
                "Output must have the same dtype as inputs");
    TORCH_CHECK(out.is_cuda(), "Output tensor must be on ROCm device");
    TORCH_CHECK(out.stride(-1) == 1,
                "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads_q, head_size_og);
    if (head_size_og % 8 != 0) {
      out = torch::empty_like(q_padded);
    }
  } else {
    out = torch::empty_like(q_padded);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  auto opts = q.options();

  auto softmax_lse = torch::empty({batch_size, num_heads_q, seqlen_q},
                                  opts.dtype(torch::kFloat32));
  torch::Tensor z;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax) {
    // TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when
    // p_dropout > 0.0");
    z = torch::empty({batch_size, num_heads_q, seqlen_q, seqlen_kv},
                     opts.dtype(torch::kUInt8));
  }

  FlashFwdBatchedParams params(batch_size, seqlen_q, seqlen_kv, num_heads_q,
                               num_heads_kv, head_size, q_padded, k_padded,
                               v_padded, out, z, softmax_lse, p_dropout,
                               softmax_scale, is_causal, return_softmax);

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state We use a custom RNG that increases the offset
  // by batch_size * nheads * 32.
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));

  int64_t counter_offset = params.b * params.h_q * 32;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  auto philox_args = gen->philox_cuda_state(counter_offset);

  if (params.is_dropout) {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);

    params.seeds = unpack(philox_args);

    // pass to backward
    auto rng_state_ptr = reinterpret_cast<uint64_t *>(rng_state.data_ptr());
    std::tie(rng_state_ptr[0], rng_state_ptr[1]) = params.seeds;
  }

  auto stream = (void *)at::cuda::getCurrentCUDAStream().stream();
//  FlashRunner flash_runner;
//  flash_runner.Run(params, stream);
  flash_run_fwd_batch(params, (void *)stream);

  torch::Tensor out_padded = out;
  if (head_size_og % 8 != 0) {
    out = out.index(
        {"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    if (out_.has_value()) {
      out_.value().copy_(out);
    }
  }

  return {out,        q_padded,    k_padded, v_padded,
          out_padded, softmax_lse, z,        rng_state};
}



void dummy_varlen_fwd() {
  throw std::runtime_error("Function 'varlen_fwd' is not available when __WMMA__ is defined.");
}

void dummy_bwd() {
  throw std::runtime_error("Function 'bwd' is not available when __WMMA__ is defined.");
}

void dummy_varlen_bwd() {
  throw std::runtime_error("Function 'varlen_bwd' is not available when __WMMA__ is defined.");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("varlen_fwd", &dummy_varlen_fwd, "Forware pass (variable length, dummy)");
    m.def("bwd", &dummy_bwd, "Backward pass (dummy)");
    m.def("varlen_bwd", &dummy_varlen_bwd, "Backward pass (variable length, dummy)");
}
