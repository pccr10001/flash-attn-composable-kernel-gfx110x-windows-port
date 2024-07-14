
#include "flash_runner.hpp"


//template <typename FlashParams>
//__declspec(dllexport) void FlashRunner::Run(FlashParams &params, void *stream)
//{
//    hipStream_t s = (hipStream_t)stream;
//    Run(params, s);
//    
//}
//
__declspec(dllexport) 
bool chk_is_xdl_supported() {
  return ck::get_device_name() == "gfx908" ||
         ck::get_device_name() == "gfx90a" ||
         ck::get_device_name() == "gfx940" ||
         ck::get_device_name() == "gfx941" || ck::get_device_name() == "gfx942";
}

__declspec(dllexport) 
bool chk_is_wmma_supported() {
  return ck::get_device_name() == "gfx1100" ||
         ck::get_device_name() == "gfx1101" ||
         ck::get_device_name() == "gfx1102";
}

__declspec(dllexport) void flash_run_fwd_batch(FlashFwdBatchedParams bp, void *s)
{
    hipStream_t stream = (hipStream_t)s;
    FlashRunner fl;
    fl.Run(bp, stream);
}

/*
__declspec(dllexport) void flash_run_fwd_group(FlashFwdGroupedParams bp, void *s)
{
    hipStream_t stream = (hipStream_t)s;
    FlashRunner fl;
    fl.Run(bp, stream);
}

__declspec(dllexport) void flash_run_bwd_batch(FlashBwdBatchedParams bp, void *s)
{
    hipStream_t stream = (hipStream_t)s;
    FlashRunner fl;
    fl.Run(bp, stream);
}

__declspec(dllexport) void flash_run_bwd_group(FlashBwdGroupedParams bp, void *s)
{
    hipStream_t stream = (hipStream_t)s;
    FlashRunner fl;
    fl.Run(bp, stream);
}
*/