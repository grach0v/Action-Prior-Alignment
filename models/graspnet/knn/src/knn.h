#pragma once

#include <torch/extension.h>
#include <vector>

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif

inline int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx) {
  TORCH_CHECK(ref.dim() == 3, "ref tensor must be of shape [B, C, N]");
  TORCH_CHECK(query.dim() == 3, "query tensor must be of shape [B, C, M]");
  TORCH_CHECK(idx.dim() == 3, "idx tensor must be of shape [B, K, M]");

  auto ref_contig = ref.contiguous();
  auto query_contig = query.contiguous();
  auto idx_contig = idx.contiguous();

  TORCH_CHECK(ref_contig.scalar_type() == at::kFloat,
              "ref tensor must be float32");
  TORCH_CHECK(query_contig.scalar_type() == at::kFloat,
              "query tensor must be float32");
  TORCH_CHECK(idx_contig.scalar_type() == at::kLong,
              "idx tensor must be int64");

  const int64_t batch = ref_contig.size(0);
  const int64_t dim = ref_contig.size(1);
  const int64_t ref_nb = ref_contig.size(2);
  const int64_t query_nb = query_contig.size(2);
  const int64_t k = idx_contig.size(1);

  TORCH_CHECK(query_contig.size(0) == batch,
              "batch dimension mismatch between ref and query tensors");
  TORCH_CHECK(query_contig.size(1) == dim,
              "channel dimension mismatch between ref and query tensors");
  TORCH_CHECK(idx_contig.size(0) == batch,
              "batch dimension mismatch between ref and idx tensors");
  TORCH_CHECK(idx_contig.size(2) == query_nb,
              "idx tensor last dimension must match number of query points");

  float* ref_ptr = ref_contig.data_ptr<float>();
  float* query_ptr = query_contig.data_ptr<float>();
  int64_t* idx_ptr = idx_contig.data_ptr<int64_t>();

  if (ref_contig.is_cuda()) {
#ifdef WITH_CUDA
    c10::cuda::CUDAGuard device_guard(ref_contig.device());

    auto dist_tensor =
        at::empty({ref_nb * query_nb}, ref_contig.options().dtype(at::kFloat));
    float* dist_ptr = dist_tensor.data_ptr<float>();
    auto stream = at::cuda::getCurrentCUDAStream();

    for (int64_t b = 0; b < batch; ++b) {
      knn_device(ref_ptr + b * dim * ref_nb, ref_nb,
                 query_ptr + b * dim * query_nb, query_nb, dim, k, dist_ptr,
                 idx_ptr + b * k * query_nb, stream.stream());
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error in knn_device: ", cudaGetErrorString(err));
#else
    TORCH_CHECK(false, "knn was compiled without CUDA support");
#endif
  } else {
    std::vector<float> dist_buf(ref_nb * query_nb);
    std::vector<int64_t> ind_buf(ref_nb);

    for (int64_t b = 0; b < batch; ++b) {
      knn_cpu(ref_ptr + b * dim * ref_nb, ref_nb,
              query_ptr + b * dim * query_nb, query_nb, dim, k,
              dist_buf.data(), idx_ptr + b * k * query_nb, ind_buf.data());
    }
  }

  if (!idx.is_contiguous()) {
    idx.copy_(idx_contig);
  }

  return 1;
}
