// strix-skeleton-serve: the shared serve shell with only the skeleton family.
//   strix-skeleton-serve --registry-root examples/skeleton_engine --registry models/registry.toml
//       --model skeleton --port 8080
#include "skeleton_engine.hpp"

#include <iostream>

int main(int argc, char** argv) {
    strix::serve::ServiceFactories factories;
    skeleton::install_session_service(factories);
    return strix::serve::run(argc, argv, factories, std::cout, std::cerr);
}
