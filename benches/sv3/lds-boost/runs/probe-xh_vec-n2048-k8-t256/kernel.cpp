#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t s) {
  uint32_t wa[4];
  float accA[1];
  float accB[1];
  float2 xa2[4];
  float2 xb2[4];
  half plo[4];
  half phi[4];
  float2 pa[1];
  float2 pb[1];
  float2 accA2[1];
  float2 accB2[1];
  #pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    float2 _;
      uint1 v_ = *(__ubuf__ uint1*)(((__ubuf__ half*)buf_dyn_shmem) + ((i * 2) + 12288));
      ((float2*)(&_))[0] = __half22float2(((half2*)(&v_))[0]);
    xa2[i] = _;
    float2 __1;
      uint1 v__1 = uint1{((__ubuf__ half*)buf_dyn_shmem)[((i * 2) + 12289)], ((__ubuf__ half*)buf_dyn_shmem)[((i * 2) + 12290)]};
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v__1))[0]);
    xb2[i] = __1;
  }
  asc_syncthreads();
  for (int32_t q = 0; q < 4; ++q) {
    accA[0] = float(0x0p+0f/*0.000000e+00*/);
    accB[0] = float(0x0p+0f/*0.000000e+00*/);
    #pragma unroll
    for (int32_t i_1 = 0; i_1 < 4; ++i_1) {
      wa[i_1] = ((__ubuf__ uint32_t*)buf_dyn_shmem)[(((i_1 * 1024) + (q * 256)) + ((int32_t)threadIdx.x))];
    }
    #pragma unroll
    for (int32_t i_2 = 0; i_2 < 4; ++i_2) {
        uint16_t v__2 = (uint16_t)(wa[i_2] & (uint32_t)65535);
      plo[i_2] = (*(half *)(&(v__2)));
        uint16_t v__3 = (uint16_t)(wa[i_2] >> (uint32_t)16);
      phi[i_2] = (*(half *)(&(v__3)));
    }
    for (int32_t c = 0; c < 4; ++c) {
        float v__4 = (float)plo[c];
        float2 __2;
        __2.x = v__4;
        __2.y = v__4;
      pa[0] = __2;
        float v__5 = (float)phi[c];
        float2 __3;
        __3.x = v__5;
        __3.y = v__5;
      pb[0] = __3;
      accA2[0] = (pa[0] * xa2[c]);
      accB2[0] = (pb[0] * xb2[c]);
        float2 v__6 = accA2[0];
        float2 v__7 = accA2[0];
      accA[0] = ((accA[0] + v__6.x) + v__7.y);
        float2 v__8 = accB2[0];
        float2 v__9 = accB2[0];
      accB[0] = ((accB[0] + v__8.x) + v__9.y);
    }
    ((__ubuf__ float*)buf_dyn_shmem)[((((s * 1024) + (q * 256)) + ((int32_t)threadIdx.x)) + 4096)] = (accA[0] + accB[0]);
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

