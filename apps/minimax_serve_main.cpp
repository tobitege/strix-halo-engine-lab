// Compatibility spelling; registry construction and ownership use the common shell.
#include "minimax/serve/session_service.hpp"
int main(int argc, char** argv) { return strix::minimax::serve::legacy_serve(argc, argv); }
