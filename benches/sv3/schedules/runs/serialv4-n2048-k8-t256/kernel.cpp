// serialv4: serial schedule with the F2F v4 recipe applied to the convert stage.
// Split the load+convert loop into a pure LDS band (uint32 words into registers),
// an asc_syncthreads() barrier, then a pure convert band (bit-cast half2 ->
// __half22float2, no UB re-read, no PRMT), then a closing barrier - exactly the
// f2f_packed_v4 window shape. x-side stays as-is. Live state unchanged:
// acc[4] + xf[8] + aw[32] (words) = 44 fp32-equivalents/thread.
#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t s) {
  float acc[4];
  float xf[8];
  float af[32];
  uint32_t aw[32];
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
  // band 1: pure LDS - stage A as uint32 words (2 words = 4 halfs per i_1,
  // same coverage as the original uint2 load)
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 8; ++i_1) {
    aw[i_1 * 2] = *((__ubuf__ uint32_t*)buf_dyn_shmem + (((i_1 >> 2) * 2048) + (((int32_t)threadIdx.x) * 8) + ((i_1 & 3) * 2)));
    aw[i_1 * 2 + 1] = *((__ubuf__ uint32_t*)buf_dyn_shmem + (((i_1 >> 2) * 2048) + (((int32_t)threadIdx.x) * 8) + ((i_1 & 3) * 2) + 1));
  }
  // align warps before the pure convert window (v4 recipe)
  asc_syncthreads();
  // band 2: pure F2F - bit-cast words to half2, packed convert, no PRMT
  // af layout unchanged: af[i_2*4 + 0..1] = convert(aw[i_2*2]),
  // af[i_2*4 + 2..3] = convert(aw[i_2*2 + 1])
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 8; ++i_2) {
    uint32_t bits0 = aw[i_2 * 2];
    uint32_t bits1 = aw[i_2 * 2 + 1];
    float2 f0;
    float2 f1;
      ((float2*)(&f0))[0] = __half22float2(((half2*)(&bits0))[0]);
      ((float2*)(&f1))[0] = __half22float2(((half2*)(&bits1))[0]);
    *(float2*)(af + (i_2 * 4)) = f0;
    *(float2*)(af + ((i_2 * 4) + 2)) = f1;
  }
  // close the measured convert window (v4 recipe)
  asc_syncthreads();
  #pragma unroll
  for (int32_t i_3 = 0; i_3 < 2; ++i_3) {
    float broadcast_var = float(0x0p+0f/*0.000000e+00*/);
    *(float2*)(acc + (i_3 * 2)) = make_float2(broadcast_var, broadcast_var);
  }
  for (int32_t k = 0; k < 8; ++k) {
    #pragma unroll
    for (int32_t i_4 = 0; i_4 < 4; ++i_4) {
      acc[i_4] = (acc[i_4] + (af[((i_4 * 8) + k)] * xf[k]));
    }
  }
  asc_syncthreads();
  #pragma unroll
  for (int32_t i_5 = 0; i_5 < 2; ++i_5) {
    *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + ((((s * 1024) + (i_5 * 512)) + (((int32_t)threadIdx.x) * 2)) + 4096)) = *(float2*)(acc + (i_5 * 2));
  }
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

#ifdef _WIN32
#define TL_EXPORT __declspec(dllexport)
#else
#define TL_EXPORT
#endif
#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];
extern "C" TL_EXPORT const char* get_last_error() { return error_buf; }
extern "C" TL_EXPORT int init() { error_buf[0] = '\0'; return 0; }
extern "C" TL_EXPORT int call(void* A, void* X, void* Out, void* stream) {
  main_kernel<<<(1 * 1 * 1), 24616, stream>>>((__gm__ half*)A, (__gm__ float*)Out, (__gm__ half*)X);
  return 0;
}
