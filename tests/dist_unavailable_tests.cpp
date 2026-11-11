// The stand-ins a platform without the pair transport links (src/dist_unavailable.cpp).
#include "strix/dist/communicator.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}
template <typename Open> bool refuses(Open open) {
    try { (void)open(); }
    catch (const std::runtime_error& error) { return std::string(error.what()).find("requires Linux") != std::string::npos; }
    return false;
}
}

int main() {
    using namespace std::chrono_literals;
    expect(refuses([] { return strix::dist::tcp_listen(29872, 10ms); }), "tcp_listen reports the requirement");
    expect(refuses([] { return strix::dist::tcp_connect("127.0.0.1", 29872, 10ms); }), "tcp_connect reports the requirement");
    expect(refuses([] { return strix::dist::usb4_host({"stream0"}); }), "usb4_host reports the requirement");
    expect(refuses([] { return strix::dist::usb4_host_auto({}); }), "usb4_host_auto reports the requirement");

    if (failures) return 1;
    std::cout << "dist stand-ins: pair links refused\n";
}
