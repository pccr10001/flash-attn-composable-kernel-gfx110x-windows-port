
# cmake .. -G Ninja -DHIP_PLATFORM=amd -DCMAKE_CXX_COMPILER_ID=Clang -D_CMAKE_HIP_DEVICE_RUNTIME_TARGET=ON -DCMAKE_CXX_COMPILER_FORCED=true -DCMAKE_HIP_ARCHITECTURES=gfx1100

cmake_minimum_required(VERSION 3.26)

project(FMHADecoderMain LANGUAGES   CXX  HIP ) 

set(CMAKE_CXX_STANDARD 20)

find_package(hip REQUIRED)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH  OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
message("PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")
message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
set(TORCH_LIB_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/../../lib")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_VERBOSE_MAKEFILE on)

set(ck_include composable_kernel/include)

file(GLOB_RECURSE SRC_DIR_LIST "src/*_gfx110x.hip") 

add_library(ck_fttn_lib SHARED 
src/device_memory.hip
src/flash_runner.hip
${SRC_DIR_LIST}
)
 

target_include_directories(ck_fttn_lib PUBLIC
src
../include/pytorch_rocm_header/include
../include/pytorch_rocm_header/include/torch/csrc/api/include
${PYTHON_INCLUDE_DIRS}
${ck_include}
composable_kernel/library/include
${HIP_INCLUDE_DIR}
)

target_link_libraries(ck_fttn_lib PUBLIC
${PYTHON_LIBRARIES}
-lc10
-lc10_cuda
-ltorch_cpu
-ltorch
-ltorch_python
#-lamdhip64
)

target_link_directories(ck_fttn_lib PUBLIC
${TORCH_LIB_PATH}
)

target_compile_options(ck_fttn_lib PUBLIC 
-Ofast
-mllvm -amdgpu-function-calls=false
-mllvm=-amdgpu-early-inline-all=true
-Wno-deprecated-declarations 
-Wno-ignored-attributes
#--save-temps
)

target_compile_definitions(ck_fttn_lib PUBLIC 
ATTN_FWD_DECODER_MAIN=1
GLIBCXX_USE_CXX11_ABI=0
_GLIBCXX_USE_CXX11_ABI=0
__CUDA_NO_HALF_OPERATORS__
__CUDA_NO_HALF_CONVERSIONS__
__HIP_PLATFORM_HCC__=1
__HIP_ROCclr__=1
USE_ROCM=1
__WMMA__
)


target_link_libraries(ck_fttn_lib PUBLIC hip::device)


link_libraries(ck_fttn_lib)
