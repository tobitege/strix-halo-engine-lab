include_guard(GLOBAL)

# Release support matrix: which compile targets are qualified (tuning and numerics measured on that hardware) and
# which are only built. docs/install.md mirrors these lists; strix-package-support-tests fails when
# include/strix/hip/arch.hpp marks a different HIP target as qualified.
set(STRIX_HIP_KNOWN_ARCHS gfx1151 gfx1200 gfx1201 gfx942 gfx950)
set(STRIX_HIP_QUALIFIED_ARCHS gfx1151)
# CUDA: sm_120 measured on an RTX PRO 6000 Blackwell (Windows 11, driver 617.14, CUDA 13.0).
set(STRIX_CUDA_QUALIFIED_ARCHS 120)

# strix_support_status(<hip|cuda> <arch> <out>): qualification for the target OS/CPU/backend/arch.
function(strix_support_status kind arch out)
    string(TOUPPER "${kind}" _kind)
    string(REGEX REPLACE "^sm_" "" _arch "${arch}")
    string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
    if(_arch IN_LIST STRIX_${_kind}_QUALIFIED_ARCHS AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND
       ((_kind STREQUAL "HIP" AND CMAKE_SYSTEM_NAME STREQUAL "Linux") OR
        (_kind STREQUAL "CUDA" AND CMAKE_SYSTEM_NAME STREQUAL "Windows")))
        set(${out} qualified PARENT_SCOPE)
    else()
        set(${out} unqualified PARENT_SCOPE)
    endif()
endfunction()

# strix_support_lines(<hip|cuda> <out> <arch>...): "arch: status" per arch, for package descriptions and the
# share/strix/support/<kind>-archs.txt receipt installed with each backend package.
function(strix_support_lines kind out)
    set(_lines "")
    foreach(_a IN LISTS ARGN)
        strix_support_status(${kind} "${_a}" _s)
        list(APPEND _lines "${_a}: ${_s}")
    endforeach()
    set(${out} "${_lines}" PARENT_SCOPE)
endfunction()
