# Shared power sampler (strix/power/sampler.hpp): host only, no device runtime, no dependency but threads.
# An independent target: every library that compiles a caller of strix::power links it by name (strix_core,
# strix_serve, strix_serve_win, strix_device_hip), so no configuration depends on another target carrying it along.
if(TARGET strix_power)
    return()
endif()
find_package(Threads REQUIRED)
add_library(strix_power STATIC ${CMAKE_CURRENT_LIST_DIR}/../src/power/sampler.cpp)
target_include_directories(strix_power PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)
target_compile_features(strix_power PUBLIC cxx_std_20)
target_link_libraries(strix_power PUBLIC Threads::Threads)
if(MSVC)
    target_compile_definitions(strix_power PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(COMMAND strix_target_warnings)
    strix_target_warnings(strix_power)
endif()
