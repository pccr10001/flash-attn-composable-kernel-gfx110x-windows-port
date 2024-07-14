# flash-attn-composable-kernel-gfx110x-windows-port
CK-Based flash attention for navi3x porting for win zluda env.

folder `ck_fattn_ker` forked and modified from:
https://github.com/ROCm/flash-attention/tree/howiejay/navi_support/csrc/flash_attn_rocm

# Build

### Step 1: Build CK library and flash attention kernel
 
need: HIP SDK 5.7.1, cmake, ninja

in `ck_fattn_ker`

```bash
mkdir build
cd build
cmake .. -G Ninja -DHIP_PLATFORM=amd -DCMAKE_CXX_COMPILER_ID=Clang -D_CMAKE_HIP_DEVICE_RUNTIME_TARGET=ON -DCMAKE_CXX_COMPILER_FORCED=true -DCMAKE_HIP_ARCHITECTURES=gfx1100
ninja
```

generate `ck_fttn_lib.dll`

### Step 2: Build python bind module

need: MSVC, cmake, cuda11.8 toolchain, ninja

in `bridge`

```
mkdir build
cd build
cmake .. -G Ninja
ninja
```

generate `ck_fttn_pyb.pyd`



