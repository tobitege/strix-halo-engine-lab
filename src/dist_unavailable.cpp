// The pair runtime on a platform without it (see strix_dist in CMakeLists.txt): every function
// that opens a link to a peer refuses. The runtime itself (dist_communicator.cpp,
// dist_gpu_direct*.cpp) is written against Linux sockets, threads, USB4 stream devices and
// DMA-BUF, and is not built here. A single-node program never opens a link; one that is asked to
// reports the requirement.
#include "strix/dist/communicator.hpp"

#include <stdexcept>
#include <string>

namespace strix::dist {
namespace {

[[noreturn]] void unavailable(const char* what) {
    throw std::runtime_error(std::string("dist transport: ") + what + " requires Linux");
}

} // namespace

std::unique_ptr<Transport> tcp_listen(std::uint16_t, std::chrono::milliseconds) { unavailable("a TCP pair link"); }
std::unique_ptr<Transport> tcp_connect(const std::string&, std::uint16_t, std::chrono::milliseconds) {
    unavailable("a TCP pair link");
}
std::unique_ptr<Transport> usb4_host(const std::vector<std::string>&, const std::vector<std::uint32_t>&) {
    unavailable("a USB4 pair link");
}
std::unique_ptr<Transport> usb4_host_auto(const Usb4AutoOptions&, Usb4Topology*) { unavailable("a USB4 pair link"); }

} // namespace strix::dist
