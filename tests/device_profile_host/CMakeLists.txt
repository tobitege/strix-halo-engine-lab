cmake_minimum_required(VERSION 3.24)
project(strix_device_profile_host LANGUAGES CXX)
enable_testing()
add_executable(strix-device-profile-host-tests ../device_profile_host_tests.cpp ../../src/device_profile.cpp)
target_include_directories(strix-device-profile-host-tests PRIVATE ../../include)
target_compile_features(strix-device-profile-host-tests PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(strix-device-profile-host-tests PRIVATE /W4 /WX)
else()
    target_compile_options(strix-device-profile-host-tests PRIVATE -Wall -Wextra -Werror -Wshadow -Wsign-conversion -Wunused-function)
endif()
add_test(NAME strix-device-profile-host-tests COMMAND strix-device-profile-host-tests)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME strix-hip-arch-caps-host-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/../hip_arch_caps_host_tests.py" --compiler "${CMAKE_CXX_COMPILER}")
add_test(NAME strix-hip-archs-default COMMAND "${CMAKE_COMMAND}" -DEXPECTED=gfx1151
    -P "${CMAKE_CURRENT_LIST_DIR}/archs_test.cmake")
add_test(NAME strix-hip-archs-multi COMMAND "${CMAKE_COMMAND}" "-DSTRIX_HIP_ARCHS=gfx1151;gfx1152" "-DEXPECTED=gfx1151;gfx1152"
    -P "${CMAKE_CURRENT_LIST_DIR}/archs_test.cmake")
add_test(NAME strix-hip-archs-legacy COMMAND "${CMAKE_COMMAND}" "-DSTRIX_HIP_ARCHITECTURES=gfx1151;gfx1152" "-DEXPECTED=gfx1151;gfx1152"
    -P "${CMAKE_CURRENT_LIST_DIR}/archs_test.cmake")
add_executable(strix-hip-arch-layer-host-tests ../hip_arch_layer_host_tests.cpp)
target_include_directories(strix-hip-arch-layer-host-tests PRIVATE ../../include)
target_compile_features(strix-hip-arch-layer-host-tests PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(strix-hip-arch-layer-host-tests PRIVATE /W4 /WX)
else()
    target_compile_options(strix-hip-arch-layer-host-tests PRIVATE -Wall -Wextra -Werror -Wshadow -Wsign-conversion)
endif()
add_test(NAME strix-hip-arch-layer-host-tests COMMAND strix-hip-arch-layer-host-tests)
add_test(NAME strix-hip-arch-target-host-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/../hip_arch_target_host_tests.py" --compiler "${CMAKE_CXX_COMPILER}")
add_test(NAME strix-hip-archs-rdna4-cdna COMMAND "${CMAKE_COMMAND}" "-DSTRIX_HIP_ARCHS=gfx1151;gfx1201;gfx942" "-DEXPECTED=gfx1151;gfx1201;gfx942"
    -P "${CMAKE_CURRENT_LIST_DIR}/archs_test.cmake")
