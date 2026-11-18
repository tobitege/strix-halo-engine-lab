#include "strix/qwen35/prefill_controls.hpp"
#include <stdexcept>
#include <string>
void require(bool value) { if (!value) throw std::runtime_error("prefill control mismatch"); }
struct Options {
    strix::cuda_core::FlashPipe flash_pipe = strix::cuda_core::FlashPipe::pad2;
    strix::cuda_core::FlashTail flash_tail = strix::cuda_core::FlashTail::split;
    bool prefill_attn8 = false;
};
int main() {
    using strix::qwen35::resolve_flash_controls;
    Options o;
    resolve_flash_controls(o, "", "", "");
    require(o.flash_pipe == strix::cuda_core::FlashPipe::pad2 && o.flash_tail == strix::cuda_core::FlashTail::split && !o.prefill_attn8);
    resolve_flash_controls(o, "swz3", "off", "on");
    require(o.flash_pipe == strix::cuda_core::FlashPipe::swz3 && o.flash_tail == strix::cuda_core::FlashTail::off && o.prefill_attn8);
    resolve_flash_controls(o, "pad2", "split", "off");
    require(o.flash_pipe == strix::cuda_core::FlashPipe::pad2 && o.flash_tail == strix::cuda_core::FlashTail::split && !o.prefill_attn8);
    for (int which = 0; which < 3; ++which) {
        bool failed = false;
        try { resolve_flash_controls(o, which == 0 ? "bad" : "swz3", which == 1 ? "bad" : "off", which == 2 ? "bad" : "on"); }
        catch (const std::runtime_error& e) { failed = std::string(e.what()).find("expects") != std::string::npos; }
        require(failed && o.flash_pipe == strix::cuda_core::FlashPipe::pad2 && o.flash_tail == strix::cuda_core::FlashTail::split && !o.prefill_attn8);
    }
}
