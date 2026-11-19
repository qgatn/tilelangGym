#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t s) {
  float accA[1];
  float accB[1];
  float e0[1];
  float e0b[1];
  float e1[1];
  float e1b[1];
  float e2[1];
  float e2b[1];
  float xfa[4];
  float xfb[4];
  float af0a[4];
  float af0b[4];
  float af1a[4];
  float af1b[4];
  float af2a[4];
  float af2b[4];
  float af3a[4];
  float af3b[4];
  #pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    float2 _;
      uint1 v_ = *(__ubuf__ uint1*)(((__ubuf__ half*)buf_dyn_shmem) + ((i * 2) + 12288));
      ((float2*)(&_))[0] = __half22float2(((half2*)(&v_))[0]);
    *(float2*)(xfa + (i * 2)) = _;
    float2 __1;
      uint1 v__1 = *(__ubuf__ uint1*)(((__ubuf__ half*)buf_dyn_shmem) + ((i * 2) + 12292));
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v__1))[0]);
    *(float2*)(xfb + (i * 2)) = __1;
  }
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 4; ++i_1) {
    af0a[i_1] = ((float)((__ubuf__ half*)buf_dyn_shmem)[((i_1 * 1024) + ((int32_t)threadIdx.x))]);
    af0b[i_1] = ((float)((__ubuf__ half*)buf_dyn_shmem)[(((i_1 * 1024) + ((int32_t)threadIdx.x)) + 4096)]);
  }
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 4; ++i_2) {
    af1a[i_2] = ((float)((__ubuf__ half*)buf_dyn_shmem)[(((i_2 * 1024) + ((int32_t)threadIdx.x)) + 256)]);
    af1b[i_2] = ((float)((__ubuf__ half*)buf_dyn_shmem)[(((i_2 * 1024) + ((int32_t)threadIdx.x)) + 4352)]);
  }
  #pragma unroll
  for (int32_t i_3 = 0; i_3 < 4; ++i_3) {
    af2a[i_3] = ((float)((__ubuf__ half*)buf_dyn_shmem)[(((i_3 * 1024) + ((int32_t)threadIdx.x)) + 512)]);
    af2b[i_3] = ((float)((__ubuf__ half*)buf_dyn_shmem)[(((i_3 * 1024) + ((int32_t)threadIdx.x)) + 4608)]);
  }
  accA[0] = float(0x0p+0f/*0.000000e+00*/);
  accB[0] = float(0x0p+0f/*0.000000e+00*/);
  e0[0] = float(0x0p+0f/*0.000000e+00*/);
  e0b[0] = float(0x0p+0f/*0.000000e+00*/);
  e1[0] = float(0x0p+0f/*0.000000e+00*/);
  e1b[0] = float(0x0p+0f/*0.000000e+00*/);
  e2[0] = float(0x0p+0f/*0.000000e+00*/);
  e2b[0] = float(0x0p+0f/*0.000000e+00*/);
  #pragma unroll
  for (int32_t i_4 = 0; i_4 < 4; ++i_4) {
    af3a[i_4] = ((float)((__ubuf__ half*)buf_dyn_shmem)[(((i_4 * 1024) + ((int32_t)threadIdx.x)) + 768)]);
    af3b[i_4] = ((float)((__ubuf__ half*)buf_dyn_shmem)[(((i_4 * 1024) + ((int32_t)threadIdx.x)) + 4864)]);
  }
  for (int32_t k = 0; k < 4; ++k) {
    accA[0] = (accA[0] + (af0a[k] * xfa[k]));
    accB[0] = (accB[0] + (af0b[k] * xfb[k]));
  }
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 1024) + ((int32_t)threadIdx.x)) + 4096)] = (accA[0] + accB[0]);
  for (int32_t k_1 = 0; k_1 < 4; ++k_1) {
    e0[0] = (e0[0] + (af1a[k_1] * xfa[k_1]));
    e0b[0] = (e0b[0] + (af1b[k_1] * xfb[k_1]));
    e1[0] = (e1[0] + (af2a[k_1] * xfa[k_1]));
    e1b[0] = (e1b[0] + (af2b[k_1] * xfb[k_1]));
    e2[0] = (e2[0] + (af3a[k_1] * xfa[k_1]));
    e2b[0] = (e2b[0] + (af3b[k_1] * xfb[k_1]));
  }
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 1024) + ((int32_t)threadIdx.x)) + 4352)] = (e0[0] + e0b[0]);
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 1024) + ((int32_t)threadIdx.x)) + 4608)] = (e1[0] + e1b[0]);
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 1024) + ((int32_t)threadIdx.x)) + 4864)] = (e2[0] + e2b[0]);
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* A, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[12288]))), (__gm__ uint8_t*)((&(X[0]))), 1, 16, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16, 16);
  for (int32_t s = 0; s < 2; ++s) {
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(A[(s * 1024)]))), 8, 2048, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4096, 2048);
    asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(256), buf_dyn_shmem, s);
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    if (s == 1) {
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    }
  }
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[4096]))), 1, 8192, static_cast<asc_store_l2_cache_mode>(4), 8192, 8192);
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
}

