// Host planner kit: predicts the Qwen prefill split over a card -> node A -> node B chain with the shared planner
// (strix/pair/chain.hpp) from declared rates, measured per-layer node costs and link rates. Prints the chosen plan
// and the best plan of every device run, with reject reasons. Prediction only: no chain route is admitted from it.
//   strix-qwen-chain-plan --rows 32768 --card-rate 4320 --node-rate 1600 --link-mbps 118 [--pair-mbps 4580]
//       [--blocks 2048,4096,8192] [--decoder card|node-a|node-b|any] [--costs node-costs.json] [--latency-ms 0.3]
//       [--card-fixed-ms 0] [--node-fixed-ms 0] [--split auto|device-only|handoff-full|layer-cut] [--cuts 24,0]
#include "strix/qwen_prefill_balance.hpp"
#include "strix/qwen_prefill_chain.hpp"
#include <cstdio>
#include <cstdlib>
#include <charconv>
#include <cmath>
#include <string>
#include <string_view>

namespace {
double number(const char* text, const char* name) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value) || !(value >= 0.0))
        throw std::invalid_argument(std::string("bad value for ") + name);
    return value;
}
template<class T> T integer(std::string_view text, const char* name) {
    T value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument(std::string("bad value for ") + name);
    return value;
}
std::vector<std::uint32_t> blocks(std::string text) {
    std::vector<std::uint32_t> out;
    std::size_t at = 0;
    while (at <= text.size()) {
        const auto comma = text.find(',', at);
        const auto part = text.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        const auto value = integer<std::uint32_t>(part, "--blocks");
        if (value < 1U || value > 65536U) throw std::invalid_argument("--blocks entries must be 1..65536");
        out.push_back(value);
        if (comma == std::string::npos) break;
        at = comma + 1U;
    }
    return out;
}
void print(const char* label, const std::vector<strix::pair::chain::Device>& devices, const strix::pair::chain::Plan& plan,
    std::uint64_t rows) {
    if (!plan.valid) {
        std::printf("%s %s..%s decoder=%s REJECT %s\n", label, devices[plan.first].name.c_str(), devices[plan.last].name.c_str(),
            devices[plan.decoder].name.c_str(), plan.reason.c_str());
        return;
    }
    std::string cuts;
    for (const auto cut : plan.cuts) cuts += (cuts.empty() ? "" : "/") + std::to_string(cut);
    std::printf("%s %s..%s decoder=%s cuts=%s block=%u x%llu wall_ms=%.1f rows_per_s=%.0f fixed_state_ms=%.1f %s\n", label,
        devices[plan.first].name.c_str(), devices[plan.last].name.c_str(), devices[plan.decoder].name.c_str(), cuts.c_str(),
        plan.block, static_cast<unsigned long long>(plan.blocks), plan.wall_ms, plan.rows_per_s(rows), plan.fixed_state_ms,
        plan.reason.c_str());
}
} // namespace

int main(int argc, char** argv) try {
    std::uint64_t rows = 32768U;
    double card_rate = 0.0, node_rate = 0.0, link_mbps = 0.0, pair_mbps = 4580.0, latency = 0.3, card_fixed = 0.0, node_fixed = 0.0;
    std::string block_text = "2048,4096,8192", decoder = "any", split_text = "auto", cut_text;
    const char* costs = nullptr;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const char* value = argv[i + 1];
        if (key == "--rows") rows = integer<std::uint64_t>(value, "--rows");
        else if (key == "--card-rate") card_rate = number(value, "--card-rate");
        else if (key == "--node-rate") node_rate = number(value, "--node-rate");
        else if (key == "--link-mbps") link_mbps = number(value, "--link-mbps");
        else if (key == "--pair-mbps") pair_mbps = number(value, "--pair-mbps");
        else if (key == "--latency-ms") latency = number(value, "--latency-ms");
        else if (key == "--card-fixed-ms") card_fixed = number(value, "--card-fixed-ms");
        else if (key == "--node-fixed-ms") node_fixed = number(value, "--node-fixed-ms");
        else if (key == "--blocks") block_text = value;
        else if (key == "--decoder") decoder = value;
        else if (key == "--costs") costs = value;
        else if (key == "--split") split_text = value;
        else if (key == "--cuts") {
            cut_text = value;
            if (cut_text.empty()) throw std::invalid_argument("bad value for --cuts");
        }
        else throw std::invalid_argument("unknown option " + key);
    }
    if (argc % 2 == 0) throw std::invalid_argument("options take one value each");
    if (rows == 0U || !(card_rate > 0.0) || (!(node_rate > 0.0) && !costs))
        throw std::invalid_argument("need --rows, --card-rate and --node-rate or --costs");
    namespace chain = strix::pair::chain;
    std::vector<chain::Device> devices;
    if (costs) {  // measured node costs (ms/row per layer, head on the last layer's owner); the card takes their shape
        const auto measured = strix::qwen_pp::load_costs(costs, rows);
        devices.push_back(strix::qwen_pp::chain_seed_device("card", card_rate, measured.layers, card_fixed));
        for (const char* name : {"node-a", "node-b"}) {
            chain::Device node{name, std::vector<double>(measured.layers.begin(), measured.layers.end()), measured.lm_head, node_fixed, 0.0};
            devices.push_back(std::move(node));
        }
    } else {
        devices.push_back(strix::qwen_pp::chain_seed_device("card", card_rate, {}, card_fixed));
        devices.push_back(strix::qwen_pp::chain_seed_device("node-a", node_rate, {}, node_fixed));
        devices.push_back(strix::qwen_pp::chain_seed_device("node-b", node_rate, {}, node_fixed));
    }
    const double link = link_mbps * 1000.0, pair = pair_mbps * 1000.0; // MB/s -> bytes/ms
    if (!std::isfinite(link) || !std::isfinite(pair)) throw std::invalid_argument("link rate out of range");
    const std::vector<chain::Link> links{{link, link, latency}, {pair, pair, 0.05}};
    chain::Request request;
    request.rows = rows;
    request.blocks = blocks(block_text);
    if (decoder == "any") request.decoders = {0U, 1U, 2U};
    else if (decoder == "card") request.decoders = {0U};
    else if (decoder == "node-a") request.decoders = {1U};
    else if (decoder == "node-b") request.decoders = {2U};
    else throw std::invalid_argument("--decoder card|node-a|node-b|any");
    if (split_text == "auto") request.split = chain::Split::auto_plan;
    else if (split_text == "device-only") request.split = chain::Split::device_only;
    else if (split_text == "handoff-full") request.split = chain::Split::handoff;
    else if (split_text == "layer-cut") request.split = chain::Split::layer_cut;
    else throw std::invalid_argument("--split auto|device-only|handoff-full|layer-cut");
    if (!cut_text.empty()) {  // "24,0": 0 = auto
        std::size_t at = 0;
        while (true) {
            const auto comma = cut_text.find(',', at);
            const auto part = cut_text.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            request.cuts.push_back(integer<std::uint32_t>(part, "--cuts"));
            if (comma == std::string::npos) break;
            at = comma + 1U;
        }
    }
    const auto choice = chain::choose(devices, links, strix::qwen_pp::chain_shape(), request);
    print("chosen", devices, choice.best, rows);
    for (const auto& plan : choice.candidates) print("candidate", devices, plan, rows);
    return choice.best.valid ? 0 : 2;
} catch (const std::exception& error) { std::fprintf(stderr, "strix-qwen-chain-plan: %s\n", error.what()); return 1; }
