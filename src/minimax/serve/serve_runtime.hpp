#pragma once
#include "session_service.hpp"
namespace strix::minimax::serve {
// Unbuilt host configurations refuse native execution explicitly.
std::unique_ptr<ServiceAssets> load_native_service(const ServePreparation&);
} // namespace strix::minimax::serve
