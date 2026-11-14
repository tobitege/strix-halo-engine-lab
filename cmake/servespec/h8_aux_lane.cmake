# Aux lane (side models: prompt scoring, image encoding, embeddings): unit choice, bounded queue and clients.
# Host only; the unit list comes from the device declaration (strix_device).
strix_servespec_serve_sources(src/serve/aux_lane.cpp)
strix_servespec_host_test(NAME strix-aux-lane-tests
    SOURCES tests/aux_lane_tests.cpp
    LIBS strix_serve strix_device)
