#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t s) {
  float2 x2[4];
  uint32_t wa[4];
  float acc[1];
  float2 pa[4];
  #pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    float2 _;
      uint1 v_ = *(__ubuf__ uint1*)(((__ubuf__ half*)buf_dyn_shmem) + ((i * 2) + 12288));
      ((float2*)(&_))[0] = __half22float2(((half2*)(&v_))[0]);
    x2[i] = _;
  }
  asc_syncthreads();
  for (int32_t q = 0; q < 4; ++q) {
    acc[0] = float(0x0p+0f/*0.000000e+00*/);
    #pragma unroll
    for (int32_t i_1 = 0; i_1 < 4; ++i_1) {
      wa[i_1] = ((__ubuf__ uint32_t*)buf_dyn_shmem)[(((i_1 * 1024) + (q * 256)) + ((int32_t)threadIdx.x))];
    }
    #pragma unroll
    for (int32_t i_2 = 0; i_2 < 4; ++i_2) {
      float2 __1;
          uint32_t v__1 = wa[i_2];
        uint1 v__2 = *(uint1 *)(&(v__1));
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v__2))[0]);
      pa[i_2] = __1;
    }
    for (int32_t c = 0; c < 4; ++c) {
        float2 v__3 = pa[c];
        float2 v__4 = x2[c];
        float2 v__5 = pa[c];
        float2 v__6 = x2[c];
      acc[0] = ((acc[0] + (v__3.x * v__4.x)) + (v__5.y * v__6.y));
    }
    ((__ubuf__ float*)buf_dyn_shmem)[((((s * 1024) + (q * 256)) + ((int32_t)threadIdx.x)) + 4096)] = acc[0];
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ uint32_t* A_t, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[12288]))), (__gm__ uint8_t*)((&(X[0]))), 1, 16, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16, 16);
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
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[4096]))), 1, 8192, static_cast<asc_store_l2_cache_mode>(4), 8192, 8192);
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
}

