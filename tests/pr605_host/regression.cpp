#include "deepseek41_quality_arms.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

using strix::deepseek41::PrefillOptions;
using strix::deepseek41::PrefillResult;
namespace quality = strix::deepseek41::quality;
#include "replay_host.hpp"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void replay_reference() {
    for (const auto inherited : {0U, 256U, 512U}) {
        PrefillOptions loaded;
        loaded.decoder_replay_rows = 512U;
        for (const auto* label : {"fast3b256", "exact", "n_l0_s1", "n_l0_s2"}) {
            const auto arm = quality::arm_precision(label);
            apply_replay(loaded, arm);
            const auto run = resolve_replay(loaded, inherited);
            const bool bounded = std::string_view(label) == "fast3b256";
            require(run.decoder_replay_rows == (bounded ? 256U : 0U), "reference/null must stay OFF under inherited replay");
            const auto executed = quality::executed_precision(arm, run);
            require(quality::arm_profile(executed) == (bounded ? "fast-prefill" : "exact"), "profile must reflect execution");
            require(resolve_replay(loaded, inherited, 8192U).decoder_replay_rows == 0U, "continuation must use exact plan");
        }
        PrefillOptions inherited_arm;
        require(resolve_replay(inherited_arm, inherited).decoder_replay_rows == inherited, "ordinary options must retain inheritance");
    }
    PrefillOptions unresolved;
    require(quality::arm_profile(unresolved) == "diagnostic", "unresolved inheritance cannot attest exact");
    const auto bounded = resolve_replay(unresolved, 256U);
    const auto executed = quality::executed_precision(quality::arm_precision("exact"), bounded);
    require(quality::arm_profile(executed) == "diagnostic", "resolved bounded execution cannot attest exact");
    unresolved.decoder_replay_rows = 512U;
    require(resolve_replay(unresolved, 256U).decoder_replay_rows == 512U, "explicit nonzero option must beat environment");
}

static void write_bytes(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("fixture write failed");
}

static void model_content() {
    const std::filesystem::path target = "pr605-target.bin", overlay = "pr605-overlay.bin", copy = "pr605-copy.bin";
    struct Cleanup {
        std::array<std::filesystem::path, 3> paths;
        ~Cleanup() { for (const auto& p : paths) { std::error_code ec; std::filesystem::remove(p, ec); } }
    } cleanup{{target, overlay, copy}};
    write_bytes(target, "metadata|tensor payload A");
    write_bytes(overlay, "overlay payload A");
    const auto original = quality::model_witness(target, overlay);
    const auto stamp = std::filesystem::last_write_time(target);
    write_bytes(target, "metadata|tensor payload B");
    std::filesystem::last_write_time(target, stamp);
    require(original != quality::model_witness(target, overlay), "same-size target edit must change witness even with restored timestamp");
    write_bytes(target, "metadata|tensor payload A");
    require(original == quality::model_witness(target, overlay), "restored content must restore witness");
    write_bytes(overlay, "overlay payload B");
    require(original != quality::model_witness(target, overlay), "effective overlay content must change witness");
    require(quality::model_witness(target, {}) != quality::model_witness(target, overlay), "overlay absence must be bound");
    write_bytes(copy, "metadata|tensor payload A");
    require(quality::model_witness(target, overlay) == quality::model_witness(copy, overlay), "paths must not change model content identity");
    bool missing_rejected = false;
    try { (void)quality::model_witness(target, "pr605-missing-overlay.bin"); }
    catch (const std::runtime_error&) { missing_rejected = true; }
    require(missing_rejected, "missing effective overlay must not attest a witness");
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected replay or model");
        if (std::string_view(argv[1]) == "replay") replay_reference();
        else if (std::string_view(argv[1]) == "model") model_content();
        else throw std::runtime_error("unknown test");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
