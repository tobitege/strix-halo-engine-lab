# Image input end to end (wave4 vision spec 2a/2d/2e): the shared bridge from a family's vision::VisionAdapter to the
# API layer (vision::ApiMediaBridge, strix_vision_api_media) and the TP2 pair frame of embedding rows
# (include/strix/pair/embed_frame.hpp, header only).
if(TARGET strix_vision_preprocess AND TARGET strix_vision_encoder)
    add_library(strix_vision_api_media STATIC "${CMAKE_SOURCE_DIR}/src/vision/api_media.cpp")
    target_include_directories(strix_vision_api_media PUBLIC "${CMAKE_SOURCE_DIR}/include")
    target_compile_features(strix_vision_api_media PUBLIC cxx_std_20)
    target_link_libraries(strix_vision_api_media PUBLIC strix_vision_preprocess strix_vision_encoder)
    strix_target_warnings(strix_vision_api_media)
endif()

strix_servespec_host_test(NAME vision-api-media
    SOURCES tests/vision_api_media_tests.cpp src/session/embed_rows.cpp
    LIBS strix_vision_api_media)
if(TARGET vision-api-media)
    target_compile_definitions(vision-api-media PRIVATE "STRIX_VISION_FIXTURE_DIR=\"${CMAKE_SOURCE_DIR}/tests/vision/fixtures\"")
    target_include_directories(vision-api-media PRIVATE "${CMAKE_SOURCE_DIR}/tests")
    set_tests_properties(vision-api-media PROPERTIES LABELS "cpu;session")
endif()

strix_servespec_host_test(NAME pair-embed-frame
    SOURCES tests/pair_embed_frame_tests.cpp src/session/embed_rows.cpp)
if(TARGET pair-embed-frame)
    set_tests_properties(pair-embed-frame PROPERTIES LABELS "cpu;session")
endif()

# --vision FILE of the DeepSeek V4.1 pair server: the bridge, the family adapter, the weight reader and the device tower.
if(TARGET strix-deepseek41-serve)
    foreach(strix_vise2e_lib strix_vision_api_media strix_deepseek41_vision strix_vision_gguf strix_vision_encoder_hip)
        if(TARGET ${strix_vise2e_lib})
            target_link_libraries(strix-deepseek41-serve PRIVATE ${strix_vise2e_lib})
        endif()
    endforeach()
endif()
