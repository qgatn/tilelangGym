#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t s) {
  float accA[1];
  float accB[1];
  float x0 = float(0x0p+0f/*0.000000e+00*/);
  float x1 = float(0x0p+0f/*0.000000e+00*/);
  float x2 = float(0x0p+0f/*0.000000e+00*/);
  float x3 = float(0x0p+0f/*0.000000e+00*/);
  float x4 = float(0x0p+0f/*0.000000e+00*/);
  float x5 = float(0x0p+0f/*0.000000e+00*/);
  float x6 = float(0x0p+0f/*0.000000e+00*/);
  float x7 = float(0x0p+0f/*0.000000e+00*/);
  uint32_t w0[1];
  uint32_t w1[1];
  uint32_t w2[1];
  uint32_t w3[1];
  half l0[1];
  half h0[1];
  half l1[1];
  half h1[1];
  half l2[1];
  half h2[1];
  half l3[1];
  half h3[1];
  x0 = float(0x0p+0f/*0.000000e+00*/);
  x1 = float(0x0p+0f/*0.000000e+00*/);
  x2 = float(0x0p+0f/*0.000000e+00*/);
  x3 = float(0x0p+0f/*0.000000e+00*/);
  x4 = float(0x0p+0f/*0.000000e+00*/);
  x5 = float(0x0p+0f/*0.000000e+00*/);
  x6 = float(0x0p+0f/*0.000000e+00*/);
  x7 = float(0x0p+0f/*0.000000e+00*/);
  x0 = ((float)((__ubuf__ half*)buf_dyn_shmem)[12288]);
  x1 = ((float)((__ubuf__ half*)buf_dyn_shmem)[12289]);
  x2 = ((float)((__ubuf__ half*)buf_dyn_shmem)[12290]);
  x3 = ((float)((__ubuf__ half*)buf_dyn_shmem)[12291]);
  x4 = ((float)((__ubuf__ half*)buf_dyn_shmem)[12292]);
  x5 = ((float)((__ubuf__ half*)buf_dyn_shmem)[12293]);
  x6 = ((float)((__ubuf__ half*)buf_dyn_shmem)[12294]);
  x7 = ((float)((__ubuf__ half*)buf_dyn_shmem)[12295]);
  asc_syncthreads();
  for (int32_t q = 0; q < 4; ++q) {
    accA[0] = float(0x0p+0f/*0.000000e+00*/);
    accB[0] = float(0x0p+0f/*0.000000e+00*/);
    w0[0] = ((__ubuf__ uint32_t*)buf_dyn_shmem)[((q * 256) + ((int32_t)threadIdx.x))];
    w1[0] = ((__ubuf__ uint32_t*)buf_dyn_shmem)[(((q * 256) + ((int32_t)threadIdx.x)) + 1024)];
    w2[0] = ((__ubuf__ uint32_t*)buf_dyn_shmem)[(((q * 256) + ((int32_t)threadIdx.x)) + 2048)];
    w3[0] = ((__ubuf__ uint32_t*)buf_dyn_shmem)[(((q * 256) + ((int32_t)threadIdx.x)) + 3072)];
      uint16_t v_ = (uint16_t)(w0[0] & (uint32_t)65535);
    l0[0] = (*(half *)(&(v_)));
      uint16_t v__1 = (uint16_t)(w0[0] >> (uint32_t)16);
    h0[0] = (*(half *)(&(v__1)));
    accA[0] = (accA[0] + (((float)l0[0]) * x0));
    accB[0] = (accB[0] + (((float)h0[0]) * x1));
      uint16_t v__2 = (uint16_t)(w1[0] & (uint32_t)65535);
    l1[0] = (*(half *)(&(v__2)));
      uint16_t v__3 = (uint16_t)(w1[0] >> (uint32_t)16);
    h1[0] = (*(half *)(&(v__3)));
    accA[0] = (accA[0] + (((float)l1[0]) * x2));
    accB[0] = (accB[0] + (((float)h1[0]) * x3));
      uint16_t v__4 = (uint16_t)(w2[0] & (uint32_t)65535);
    l2[0] = (*(half *)(&(v__4)));
      uint16_t v__5 = (uint16_t)(w2[0] >> (uint32_t)16);
    h2[0] = (*(half *)(&(v__5)));
    accA[0] = (accA[0] + (((float)l2[0]) * x4));
    accB[0] = (accB[0] + (((float)h2[0]) * x5));
      uint16_t v__6 = (uint16_t)(w3[0] & (uint32_t)65535);
    l3[0] = (*(half *)(&(v__6)));
      uint16_t v__7 = (uint16_t)(w3[0] >> (uint32_t)16);
    h3[0] = (*(half *)(&(v__7)));
    accA[0] = (accA[0] + (((float)l3[0]) * x6));
    accB[0] = (accB[0] + (((float)h3[0]) * x7));
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

