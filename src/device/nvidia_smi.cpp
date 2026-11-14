#include "strix/device/gate.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#define STRIX_POPEN _popen
#define STRIX_PCLOSE _pclose
#define STRIX_NULL_DEVICE "NUL"
#else
#define STRIX_NULL_DEVICE "/dev/null"
#define STRIX_POPEN popen
#define STRIX_PCLOSE pclose
#endif

namespace strix::device {
namespace {
std::string trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) text.remove_suffix(1);
    return std::string(text);
}
} // namespace

std::vector<Visible> parse_nvidia_smi(std::string_view csv) {
    std::vector<Visible> out;
    while (!csv.empty()) {
        const auto end = csv.find('\n');
        auto line = csv.substr(0, end);
        csv.remove_prefix(end == std::string_view::npos ? csv.size() : end + 1);
        if (trim(line).empty()) continue;
        // index, uuid, pci.bus_id, memory.total [MiB], name (the name is last: it is the only free-text field)
        std::string field[5];
        for (int i = 0; i < 5; ++i) {
            const auto comma = i < 4 ? line.find(',') : std::string_view::npos;
            if (i < 4 && comma == std::string_view::npos) throw std::invalid_argument("nvidia-smi: unexpected line " + std::string(line));
            field[i] = trim(line.substr(0, comma));
            if (i < 4) line.remove_prefix(comma + 1);
        }
        Visible v;
        v.vendor = Vendor::nvidia;
        try {
            std::size_t used = 0;
            v.ordinal = std::stoi(field[0], &used);
            if (used != field[0].size() || v.ordinal < 0) throw std::invalid_argument("index");
            v.uuid = normalise_uuid(field[1]);
            v.pci = normalise_pci(field[2]);
            v.memory_bytes = std::stoull(field[3], &used) << 20U;
            if (used != field[3].size()) throw std::invalid_argument("memory");
        } catch (const std::exception& e) {
            throw std::invalid_argument("nvidia-smi: unexpected line (" + std::string(e.what()) + ")");
        }
        v.name = field[4];
        out.push_back(std::move(v));
    }
    return out;
}

std::optional<std::vector<Visible>> query_nvidia_smi() {
    std::FILE* pipe = STRIX_POPEN("nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.total,name --format=csv,noheader,nounits 2>" STRIX_NULL_DEVICE, "r");
    if (!pipe) return std::nullopt;
    std::string text;
    char buffer[512];
    while (std::fgets(buffer, sizeof buffer, pipe)) text += buffer;
    if (STRIX_PCLOSE(pipe) != 0) return std::nullopt;
    try {
        auto devices = parse_nvidia_smi(text);
        if (devices.empty()) return std::nullopt;
        return devices;
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

void set_process_environment(const char* name, const std::string& value) {
#ifdef _WIN32
    if (_putenv_s(name, value.c_str()) != 0) throw Refusal(std::string("device gate: refusing to start: cannot set ") + name);
#else
    if (setenv(name, value.c_str(), 1) != 0) throw Refusal(std::string("device gate: refusing to start: cannot set ") + name);
#endif
}

} // namespace strix::device
