#include "rotary_mock.hpp"
#include "../apps/qwen_effective_config.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Env {
    std::string key, old;
    bool had;
    static void set(const char* key, const char* value) {
#ifdef _WIN32
        _putenv_s(key, value ? value : "");
#else
        if (value) setenv(key, value, 1); else unsetenv(key);
#endif
    }
    Env(const char* k, const char* value) : key(k), had(std::getenv(k) != nullptr) {
        if (had) old = std::getenv(k);
        set(k, value);
    }
    ~Env() { set(key.c_str(), had ? old.c_str() : nullptr); }
};
strix::decode_cli::Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return strix::decode_cli::parse_options(static_cast<int>(argv.size()), argv.data());
}
void rejects(std::vector<std::string> args, std::string_view message) {
    try { (void)parse(std::move(args)); }
    catch (const std::invalid_argument& e) {
        require(std::string_view(e.what()).find(message) != std::string_view::npos, e.what());
        return;
    }
    throw std::runtime_error("request unexpectedly accepted");
}
}
int main() {
    using namespace strix;
    Env yarn("STRIX_QWEN_ROPE_YARN", nullptr), baseline("STRIX_BASELINE", nullptr);
    const auto native = RotaryDescriptor::native();
    require(native.canonical_bytes().empty() && native.table_bytes() == 0, "native is no payload/table");
    for (const auto hash : {0ULL, 1ULL, 5195923158154663034ULL})
        require(append_rotary_identity(hash, native) == hash, "native append changes no identity bytes");
    const auto two = RotaryDescriptor::yarn(2), four = RotaryDescriptor::yarn(4);
    require(two.effective_max() == 524288 && four.effective_max() == 1048576, "effective extents");
    require(two.sha256() != four.sha256() && two.sha256() == RotaryDescriptor::yarn(2).sha256(), "stable distinct identities");
    require(two.rotary_dim == 64 && two.index_rotary_dim == 64, "main/index geometry");
    require(two.attention_factor() == 1.0693147180559945 && four.attention_factor() == 1.1386294361119890, "exact double amplitudes");
    require(parse({"decoder", "model.gguf", "1"}).rotary == native, "unset native");
    require(parse({"decoder", "model.gguf", "1", "--rope-yarn", "off"}).rotary == native, "explicit off native");
    for (const auto value : {"0", "1", "3", "NaN", "-2", "2.0", "2.5", "", "02"})
        rejects({"decoder", "model.gguf", "1", "--rope-yarn", value}, "must be off, 2 or 4");
    rejects({"decoder", "model.gguf", "1", "--rope-yarn", "2"}, "requires explicit --context");
    for (const auto context : {"524289"})
        rejects({"decoder", "model.gguf", "1", "--rope-yarn", "2", "--context", context}, "0 < context");
    rejects({"decoder", "model.gguf", "1", "--rope-yarn", "4", "--context", "1048577"}, "0 < context");
    // Storage capacity never opts into experimental rotary arithmetic, on CLI or HTTP.
    rejects({"decoder", "model.gguf", "1", "--rope-yarn", "off", "--context", "262145", "--unqualified-context"}, "native rotary context");
    for (const auto context : {"262145", "300000", "524288", "524289", "1048576"}) {
        const auto factor = std::stoull(context) <= 524288U ? "2" : "4";
        for (bool http : {false, true}) {
            std::vector<std::string> args{"decoder", "model.gguf", "1", "--context", context};
            if (http) args.push_back("--http-serve");
            rejects(std::move(args), std::string("--rope-yarn ") + factor);
        }
        require(parse({"decoder", "model.gguf", "1", "--context", context, "--rope-yarn", factor}).rotary.factor() ==
            std::stoul(factor), "explicit opt-in preserves supported extent");
    }
    require(parse({"decoder", "model.gguf", "1", "--context", "262144"}).rotary == native, "native up to 262144");
    rejects({"decoder", "model.gguf", "1", "--context", "1048577"}, "native rotary context");
    {   // No --context: a known prompt above native refuses with an actionable recommendation.
        std::vector<std::string> args{"decoder", "model.gguf"};
        args.insert(args.end(), 262144U, "1");
        args.insert(args.end(), {"--generate", "1"});
        auto off = args; off.insert(off.end(), {"--rope-yarn", "off"});
        rejects(std::move(args), "--rope-yarn 2 --context 262145");
        rejects(std::move(off), "native rotary context");
        // 262,143 + 1 generated token fits native: it keeps native RoPE (bits) and its automatic context.
        std::vector<std::string> fits{"decoder", "model.gguf"};
        fits.insert(fits.end(), 262143U, "1");
        fits.insert(fits.end(), {"--generate", "1"});
        const auto n = parse(std::move(fits));
        require(n.rotary == native && !n.context_explicit, "a prompt native admits stays native");
        // Recommend the smallest covering factor without selecting it.
        std::vector<std::string> band{"decoder", "model.gguf"};
        band.insert(band.end(), 524287U, "1");
        band.insert(band.end(), {"--generate", "1"});
        rejects(std::move(band), "--rope-yarn 2 --context 524288");
    }
    require(recommended_qwen_rope_yarn(262144) == 0U && recommended_qwen_rope_yarn(262145) == 2U &&
        recommended_qwen_rope_yarn(524289) == 4U && recommended_qwen_rope_yarn(1048577) == 0U, "diagnostic factor bands");
    rejects({"decoder", "model.gguf", "1", "--rope-yarn", "2", "--context", "524288", "--baseline"}, "native rotary context");
    require(parse({"decoder","model.gguf","1","--rope-yarn","2","--context","8192","--baseline"}).rotary == native,
        "baseline suppresses CLI YaRN before YaRN context validation");
    require(validate_qwen_rotary_request(2,false,8192,true) == native,
        "descriptor master suppression precedes YaRN validation");
    for (const auto factor : {"2", "4"}) {
        Env env("STRIX_QWEN_ROPE_YARN", factor);
        require(parse({"decoder", "model.gguf", "1", "--context", "524288"}).rotary.factor() == std::stoul(factor), "environment opt-in accepted");
        require(parse({"decoder", "model.gguf", "1", "--rope-yarn", factor, "--context", "524288"}).rotary.factor() == std::stoul(factor), "equal opt-in accepted");
        require(parse({"decoder", "model.gguf", "1", "--rope-yarn", "off"}).rotary == native, "CLI off overrides environment");
        rejects({"decoder", "model.gguf", "1", "--rope-yarn", "off", "--context", "524288"}, "native rotary context");
        rejects({"decoder", "model.gguf", "1", "--rope-yarn", factor == std::string_view("2") ? "4" : "2", "--context", "524288"}, "conflicting");
        for (bool env_master : {false,true}) {
            Env master("STRIX_BASELINE",env_master ? "1" : nullptr);
            auto args = std::vector<std::string>{"decoder","model.gguf","1","--context","8192"};
            if (!env_master) args.push_back("--baseline");
            const auto o = parse(args);
            require(o.rotary == native && !o.suppressed_yarn_request.empty(), "baseline suppresses inherited YaRN before context validation");
            decode_cli::EffectiveConfig c; c.options = o;
            require(decode_cli::effective_config_json(c).find("suppressed_yarn_request") != std::string::npos &&
                decode_cli::effective_config_notice(c).find("suppressed YaRN") != std::string::npos, "baseline reports suppressed request in JSON and stderr notice");
        }
    }
    {
        Env env("STRIX_QWEN_ROPE_YARN", "NaN");
        rejects({"decoder", "model.gguf", "1"}, "must be off, 2 or 4");
    }
    for(const auto c:{1U,32768U,131072U,262144U,262145U,524288U})
        require(validate_qwen_rotary_request(2,true,c,false)==two,"explicit short/cross-native f2 geometry");
    require(validate_qwen_rotary_request(4,true,1048576,false)==four,"f4 final extent");
    rejects({"decoder","model.gguf","1","--rope-yarn","2","--context","300000","--http-serve"},
        "YaRN on HTTP serving not supported");
    rejects({"decoder","model.gguf","1","2","--rope-yarn","2","--context","300000","--teacher-forced-score","--generate","1"},
        "YaRN with teacher scoring not yet qualified");
    require(parse({"decoder","model.gguf","1","--rope-yarn=4","--context","1048576","--print-effective-config"}).print_effective_config,
        "inspection reaches host admission");
    decode_cli::EffectiveConfig config;
    require(decode_cli::effective_config_json(config).starts_with("{\"rotary\":" + native.json() + ","),
        "inspection schema deliberately adds native rotary status");
    require(RotaryIdentity{}.json() == "{}" && RotaryIdentity{}.canonical_bytes().empty(),
        "native numerical identity serialization remains empty");
    std::array<ModelContract,3> models;
    models[1].qwen_rotary_table = mock_rotary(2);
    models[2].qwen_rotary_table = mock_rotary(4);
    for (unsigned i=0;i<3U;++i) for(unsigned j=0;j<3U;++j)
        require(same_rotary_model(models[i],models[j]) == (i==j),
            "QSA cohort, GDN island and MTP model predicate distinguishes native/2/4");
    for (const auto& producer : models) for (const auto& draft : models) {
        bool refused=false;
        try { require_rotary_producer_identity(producer.rotary_identity(),draft.rotary_identity()); }
        catch(const std::invalid_argument&) { refused=true; }
        require(refused == (producer.rotary_identity()!=draft.rotary_identity()), "target producer native/2/4 cross-product admission");
    }
    auto same = models[1]; same.qwen_rotary_table = mock_rotary(2);
    require(same_rotary_model(same,models[1]), "eager identity compares payload rather than pointer");
    same.qwen_rotary_table = mock_rotary(2,2);
    bool producer_refused=false;
    try { require_rotary_producer_identity(same.rotary_identity(),models[1].rotary_identity()); }
    catch(const std::invalid_argument&) { producer_refused=true; }
    require(producer_refused,"target producer same-factor different-payload refusal");
    require(!same_rotary_model(same,models[1]), "coefficient digest changes numerical identity");
    require(models[1].rotary_identity().json().find("table_sha256\":\"01") != std::string::npos,
        "resolved receipt carries actual coefficient digest");
    for(unsigned i=0;i<models.size();++i) {
        bool refused=false;
        try { require_qwen_rotary_device_payload(models[i].rotary_identity(),nullptr); }
        catch(const std::invalid_argument&) {refused=true;}
        require(refused==(i!=0U),"resolved non-native identity without upload cannot enter a reader");
        require_qwen_rotary_device_payload(models[i].rotary_identity(),models[i].qwen_rotary_table.get());
    }
    for (const int active : {0,1}) {
        unsigned queries = 0;
        const auto query = [&](unsigned stream) {
            ++queries;
            require(stream == 3U, "only a real stream may enter device query");
            return 1 - active;
        };
        // HIP's null, legacy (1), and per-thread (2) tokens must never be queried.
        for (const unsigned stream : {0U,1U,2U})
            require(qwen_rotary_reader_device(stream,1U,2U,active,query) == active && queries == 0U,
                "default rotary reader uses current device without a stream API");
        require(qwen_rotary_reader_device(3U,1U,2U,active,query) == 1-active && queries == 1U,
            "real rotary reader preserves stream device query");
        bool refused = false;
        try { require_qwen_rotary_reader_device(active,active,
            qwen_rotary_reader_device(3U,1U,2U,active,query)); }
        catch(const std::invalid_argument&) { refused = true; }
        require(refused, "real foreign stream remains rejected");
    }
    for (const auto allocation : {-1,0,1}) for (const auto active : {0,1}) for (const auto stream : {0,1}) {
        bool refused=false;
        try { require_qwen_rotary_reader_device(allocation,active,stream); }
        catch(const std::invalid_argument&) { refused=true; }
        require(refused != (allocation>=0 && allocation==active && allocation==stream), "rotary device/stream admission matrix");
    }
    for(bool selection : {false,true}) for(bool completion : {false,true}) {
        unsigned drains=0, frees=0;
        const bool retired=retire_qwen_rotary_allocation([&]{return selection;},[&]{++drains;return completion;},[&]{++frees;});
        require(retired==(selection && completion) && frees==unsigned(retired) && drains==unsigned(selection),
            "failed device selection/drain quarantines allocation without freeing");
    }
    std::cout << "rotary descriptor/parser tests passed\n";
}
