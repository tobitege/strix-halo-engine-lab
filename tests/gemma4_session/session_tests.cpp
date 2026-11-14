#include <exception>
#include <iostream>
int gemma4_session_replay(int, char**);
int main(int argc, char** argv) {
    try { return gemma4_session_replay(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
