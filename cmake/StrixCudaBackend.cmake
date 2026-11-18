# One CUDA backend under src/cuda, the same layering as the HIP side:
#   src/cuda/core/          model-agnostic device helpers and kernels (PTX wrappers, reductions, trellis decode,
#                           the gfx1151-identical math tables)
#   src/cuda/arch/          capability queries; arch/<sm>/ holds architecture specialisations, chosen by capability
#   src/cuda/models/<m>/    what is truly model-specific (ds41, qwen, glm, minimax)
#   include/strix/cuda/     public headers of the backend
# Included by the DeepSeek configuration (CMakeLists.txt) and the Qwen-only one (cmake/QwenCuda.cmake).
include_guard(GLOBAL)
get_filename_component(_strix_cuda_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(STRIX_CUDA_ROOT "${_strix_cuda_root}/src/cuda")

# Header-only part of the core (src/cuda/core/*.cuh, src/cuda/arch/*.hpp, include/strix/cuda/*): every CUDA model
# library links it.
add_library(strix_cuda_core INTERFACE)
target_include_directories(strix_cuda_core INTERFACE "${_strix_cuda_root}/include")

# Core sources compiled into the model library that uses them, so they keep that library's exact arithmetic flags.
set(STRIX_CUDA_CORE_AMD_MATH_SOURCES "${STRIX_CUDA_ROOT}/core/amd_math_tables.cu")
set(STRIX_CUDA_CORE_TRELLIS_SOURCES "${STRIX_CUDA_ROOT}/core/trellis.cu")
set(STRIX_CUDA_CORE_DENSE_SOURCES "${STRIX_CUDA_ROOT}/core/dense/q8_a8_imma.cu")
