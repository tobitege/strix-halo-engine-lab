# Prompt compression fragments live in cmake/pflash/<name>.cmake and use the helpers of this hook.
# They are included in name order; adding one needs no edit here.
file(GLOB strix_pflash_fragments CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/../pflash/*.cmake")
list(SORT strix_pflash_fragments)
foreach(strix_pflash_fragment IN LISTS strix_pflash_fragments)
    include("${strix_pflash_fragment}")
endforeach()
