# Test-only adapter injected unchanged into both immediately preceding and candidate sources.
include_guard(GLOBAL)
get_filename_component(WFMT_CAPTURE_KIT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_executable(strix-wfmt-legacy-capture
    "${WFMT_CAPTURE_KIT}/tests/wfmt/legacy_capture.cpp"
    "${WFMT_CAPTURE_KIT}/tests/wfmt/compare.cpp"
    "${CMAKE_SOURCE_DIR}/src/gemma4/model/safetensors.cpp"
    "${CMAKE_SOURCE_DIR}/src/vision/gguf_weights.cpp")
target_include_directories(strix-wfmt-legacy-capture PRIVATE
    "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/src" "${WFMT_CAPTURE_KIT}/tests/wfmt")
target_compile_features(strix-wfmt-legacy-capture PRIVATE cxx_std_20)
target_compile_definitions(strix-wfmt-legacy-capture PRIVATE STRIX_WFMT_CAPTURE=1)
target_link_libraries(strix-wfmt-legacy-capture PRIVATE strix-weight)
if(UNIX)
    target_sources(strix-wfmt-legacy-capture PRIVATE
        "${CMAKE_SOURCE_DIR}/src/glm/load/glm_rank_plan.cpp"
        "${CMAKE_SOURCE_DIR}/src/minimax/engine/mm_pack.cpp")
    target_link_libraries(strix-wfmt-legacy-capture PRIVATE
        strix_minimax_model_description strix_minimax_assets strix_core)
endif()
if(MSVC)
    target_compile_options(strix-wfmt-legacy-capture PRIVATE /fp:strict /D_CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(strix-wfmt-legacy-capture PRIVATE -ffp-contract=off)
endif()
