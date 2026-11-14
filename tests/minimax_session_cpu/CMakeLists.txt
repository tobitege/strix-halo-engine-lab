cmake_minimum_required(VERSION 3.24)
project(minimax_session_cpu LANGUAGES CXX)
find_package(Threads REQUIRED)
set(ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
include("${ROOT}/cmake/StrixSession.cmake")
add_executable(strix-minimax-session-tests ../minimax_session_cache_tests.cpp
    ${ROOT}/src/minimax/session/session_cache.cpp
    ${ROOT}/src/minimax/model_description.cpp ${ROOT}/src/model_description.cpp
    ${ROOT}/src/minimax/model_assets.cpp ${ROOT}/src/serve/json.cpp)
target_include_directories(strix-minimax-session-tests PRIVATE ${ROOT}/include ${ROOT}/src)
if(NOT WIN32)
 target_sources(strix-minimax-session-tests PRIVATE ${ROOT}/src/minimax/engine/mm_pack.cpp)
endif()
target_compile_features(strix-minimax-session-tests PRIVATE cxx_std_20)
target_link_libraries(strix-minimax-session-tests PRIVATE strix_session)
if(MSVC)
 # The existing non-Linux loader refusal and deliberate throw-only startup
 # fixture are statically unreachable under MSVC. Retain /WX for other warnings
 # and retain the real platform omissions and all runtime assertions.
 target_compile_options(strix-minimax-session-tests PRIVATE /W4 /WX /wd4702)
else()
 target_compile_options(strix-minimax-session-tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
endif()
enable_testing()
add_test(NAME strix-minimax-session-tests COMMAND strix-minimax-session-tests)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
 add_test(NAME minimax-session-gate-helpers COMMAND ${Python3_EXECUTABLE} ${ROOT}/tests/minimax_session_cache_gate_tests.py)
 set_tests_properties(minimax-session-gate-helpers PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
 add_test(NAME minimax-session-phasing COMMAND ${Python3_EXECUTABLE} ${ROOT}/tests/minimax_session_phasing_tests.py)
 add_test(NAME minimax-session-default-observer COMMAND ${Python3_EXECUTABLE} ${ROOT}/tests/minimax_session_default_probe_tests.py)
endif()

add_executable(strix-minimax-defaults-tests ../minimax_defaults_tests.cpp ${ROOT}/src/minimax/engine/mm_defaults.cpp)
target_include_directories(strix-minimax-defaults-tests PRIVATE ${ROOT}/src ${ROOT}/include)
target_compile_features(strix-minimax-defaults-tests PRIVATE cxx_std_20)
add_test(NAME strix-minimax-defaults-tests COMMAND strix-minimax-defaults-tests ${ROOT}/tests/fixtures/minimax_production_receipt.txt)

add_executable(strix-minimax-session-api-tests ../minimax_session_api_tests.cpp ${ROOT}/src/serve/json.cpp)
target_include_directories(strix-minimax-session-api-tests PRIVATE ${ROOT}/include ${ROOT}/src)
target_compile_features(strix-minimax-session-api-tests PRIVATE cxx_std_20)
add_test(NAME strix-minimax-session-api-tests COMMAND strix-minimax-session-api-tests)

# Keep checks active under optimized configurations as in the shared projects.
foreach(target strix-minimax-session-tests strix-minimax-defaults-tests strix-minimax-session-api-tests)
 if(MSVC)
  target_compile_options(${target} PRIVATE /EHsc /utf-8 /UNDEBUG)
 else()
  target_compile_options(${target} PRIVATE -UNDEBUG)
 endif()
endforeach()
