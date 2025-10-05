cmake_minimum_required (VERSION 3.21)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_ARCH "-m64")
set(CMAKE_BUILD_TYPE Release)

project ("ConvByCluster" LANGUAGES CXX)

if (MSVC)
  # suppress C4624
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4624")
endif()

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED) 

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
message("PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")
message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message("TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

include_directories(${PYTHON_INCLUDE_DIRS})
link_directories("../ck_fattn_ker/build")
add_library(ck_fttn_pyb SHARED "main.cpp")

SET_TARGET_PROPERTIES(ck_fttn_pyb PROPERTIES SUFFIX .pyd)

target_link_libraries(ck_fttn_pyb 
  ${PYTHON_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  ck_fttn_lib.so
)

target_compile_definitions(ck_fttn_pyb PRIVATE TORCH_EXTENSION_NAME=ck_fttn_pyb)

