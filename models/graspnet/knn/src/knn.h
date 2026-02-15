#pragma once
#include "cpu/vision.h"
#include <c10/cuda/CUDAStream.h>

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif



int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{

    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data_ptr<float>();
    float *query_dev = query.data_ptr<float>();
    long *idx_dev = idx.data_ptr<long>();




  if (ref.is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    float *dist_dev = nullptr;
    auto alloc_status = cudaMalloc((void **)&dist_dev, ref_nb * query_nb * sizeof(float));
    TORCH_CHECK(alloc_status == cudaSuccess, "knn cudaMalloc failed: ", cudaGetErrorString(alloc_status));

    for (int b = 0; b < batch; b++)
    {
    // knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
    //   dist_dev, idx_dev + b * k * query_nb, THCState_getCurrentStream(state));
      knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
    }
    auto free_status = cudaFree(dist_dev);
    TORCH_CHECK(free_status == cudaSuccess, "knn cudaFree failed: ", cudaGetErrorString(free_status));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "error in knn: ", cudaGetErrorString(err));
    return 1;
#else
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }


    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;

}
