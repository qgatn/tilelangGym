#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t s) {
  float acc[1];
  float xf[8];
  float af0[8];
  float af1[8];
  float af2[8];
  float af3[8];
  #pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    half x_ub_local_cast[4];
    *(uint2*)(x_ub_local_cast + 0) = *(__ubuf__ uint2*)(((__ubuf__ half*)buf_dyn_shmem) + ((i * 4) + 12288));
    for (int32_t vec = 0; vec < 2; ++vec) {
      float2 _;
        uint1 v_ = *(uint1*)(x_ub_local_cast + (vec * 2));
        ((float2*)(&_))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(xf + ((i * 4) + (vec * 2))) = _;
    }
  }
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 2; ++i_1) {
    half a_ub_local_cast_1[4];
    *(uint2*)(a_ub_local_cast_1 + 0) = *(__ubuf__ uint2*)(((__ubuf__ half*)buf_dyn_shmem) + ((((int32_t)threadIdx.x) * 8) + (i_1 * 4)));
    for (int32_t vec_1 = 0; vec_1 < 2; ++vec_1) {
      float2 __1;
        uint1 v__1 = *(uint1*)(a_ub_local_cast_1 + (vec_1 * 2));
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v__1))[0]);
      *(float2*)(af0 + ((i_1 * 4) + (vec_1 * 2))) = __1;
    }
  }
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 2; ++i_2) {
    half a_ub_local_cast_2[4];
    *(uint2*)(a_ub_local_cast_2 + 0) = *(__ubuf__ uint2*)(((__ubuf__ half*)buf_dyn_shmem) + (((((int32_t)threadIdx.x) * 8) + (i_2 * 4)) + 2048));
    for (int32_t vec_2 = 0; vec_2 < 2; ++vec_2) {
      float2 __2;
        uint1 v__2 = *(uint1*)(a_ub_local_cast_2 + (vec_2 * 2));
        ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__2))[0]);
      *(float2*)(af1 + ((i_2 * 4) + (vec_2 * 2))) = __2;
    }
  }
  #pragma unroll
  for (int32_t i_3 = 0; i_3 < 2; ++i_3) {
    half a_ub_local_cast_3[4];
    *(uint2*)(a_ub_local_cast_3 + 0) = *(__ubuf__ uint2*)(((__ubuf__ half*)buf_dyn_shmem) + (((((int32_t)threadIdx.x) * 8) + (i_3 * 4)) + 4096));
    for (int32_t vec_3 = 0; vec_3 < 2; ++vec_3) {
      float2 __3;
        uint1 v__3 = *(uint1*)(a_ub_local_cast_3 + (vec_3 * 2));
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__3))[0]);
      *(float2*)(af2 + ((i_3 * 4) + (vec_3 * 2))) = __3;
    }
  }
  acc[0] = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t kb = 0; kb < 4; ++kb) {
    half a_ub_local_cast_4[2];
    *(uint1*)(a_ub_local_cast_4 + 0) = *(__ubuf__ uint1*)(((__ubuf__ half*)buf_dyn_shmem) + (((((int32_t)threadIdx.x) * 8) + (kb * 2)) + 6144));
    float2 __4;
      uint1 v__4 = *(uint1*)(a_ub_local_cast_4 + 0);
      ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__4))[0]);
    *(float2*)(af3 + (kb * 2)) = __4;
    for (int32_t k = 0; k < 2; ++k) {
      acc[0] = (acc[0] + (af0[((kb * 2) + k)] * xf[((kb * 2) + k)]));
    }
  }
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 1024) + ((int32_t)threadIdx.x)) + 4096)] = acc[0];
  acc[0] = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t k_1 = 0; k_1 < 8; ++k_1) {
    acc[0] = (acc[0] + (af1[k_1] * xf[k_1]));
  }
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 1024) + ((int32_t)threadIdx.x)) + 4352)] = acc[0];
  acc[0] = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t k_2 = 0; k_2 < 8; ++k_2) {
    acc[0] = (acc[0] + (af2[k_2] * xf[k_2]));
  }
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 1024) + ((int32_t)threadIdx.x)) + 4608)] = acc[0];
  acc[0] = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t k_3 = 0; k_3 < 8; ++k_3) {
    acc[0] = (acc[0] + (af3[k_3] * xf[k_3]));
  }
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 1024) + ((int32_t)threadIdx.x)) + 4864)] = acc[0];
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* A, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[12288]))), (__gm__ uint8_t*)((&(X[0]))), 1, 16, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16, 16);
  for (int32_t s = 0; s < 2; ++s) {
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(A[(s * 8192)]))), 1, 16384, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16384, 16384);
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

