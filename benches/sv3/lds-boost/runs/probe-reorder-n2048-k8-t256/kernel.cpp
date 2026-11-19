#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t s) {
  float xfa[4];
  float xfb[4];
  float accA[1];
  float accB[1];
  for (int32_t q = 0; q < 4; ++q) {
    for (int32_t c = 0; c < 4; ++c) {
      uint32_t w = ((__ubuf__ uint32_t*)buf_dyn_shmem)[(((c * 1024) + (q * 256)) + ((int32_t)threadIdx.x))];
        uint16_t v_ = (uint16_t)(w & (uint32_t)65535);
      ((__ubuf__ float*)buf_dyn_shmem)[((((c * 1024) + (q * 256)) + ((int32_t)threadIdx.x)) + 4096)] = ((float)(*(half *)(&(v_))));
        uint16_t v__1 = (uint16_t)(w >> (uint32_t)16);
      ((__ubuf__ float*)buf_dyn_shmem)[((((c * 1024) + (q * 256)) + ((int32_t)threadIdx.x)) + 8192)] = ((float)(*(half *)(&(v__1))));
    }
  }
  asc_syncthreads();
  #pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    xfa[i] = ((float)((__ubuf__ half*)buf_dyn_shmem)[((i * 2) + 28672)]);
    xfb[i] = ((float)((__ubuf__ half*)buf_dyn_shmem)[((i * 2) + 28673)]);
  }
  asc_syncthreads();
  for (int32_t q_1 = 0; q_1 < 4; ++q_1) {
    accA[0] = float(0x0p+0f/*0.000000e+00*/);
    accB[0] = float(0x0p+0f/*0.000000e+00*/);
    for (int32_t c_1 = 0; c_1 < 4; ++c_1) {
      accA[0] = (accA[0] + (((__ubuf__ float*)buf_dyn_shmem)[((((c_1 * 1024) + (q_1 * 256)) + ((int32_t)threadIdx.x)) + 4096)] * xfa[c_1]));
      accB[0] = (accB[0] + (((__ubuf__ float*)buf_dyn_shmem)[((((c_1 * 1024) + (q_1 * 256)) + ((int32_t)threadIdx.x)) + 8192)] * xfb[c_1]));
    }
    ((__ubuf__ float*)buf_dyn_shmem)[((((s * 1024) + (q_1 * 256)) + ((int32_t)threadIdx.x)) + 12288)] = (accA[0] + accB[0]);
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ uint32_t* A_t, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[28672]))), (__gm__ uint8_t*)((&(X[0]))), 1, 16, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16, 16);
  for (int32_t s = 0; s < 2; ++s) {
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ uint32_t*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(A_t[(s * 1024)]))), 4, 4096, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 8192, 4096);
    asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(256), buf_dyn_shmem, s);
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    if (s == 1) {
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    }
  }
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[12288]))), 1, 8192, static_cast<asc_store_l2_cache_mode>(4), 8192, 8192);
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
}

