#include "strix/device/gate.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>

namespace strix::device {
namespace {
std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

std::string lower(std::string_view text) {
    std::string out(text);
    for (auto& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::string shown_uuid(Vendor vendor, const std::string& uuid) {
    if (uuid.empty()) return "-";
    return vendor == Vendor::nvidia ? "GPU-" + uuid : uuid;
}

std::string describe(const Visible& v) {
    return "[" + std::to_string(v.ordinal) + "] \"" + v.name + "\" uuid=" + shown_uuid(v.vendor, v.uuid) +
           " pci=" + (v.pci.empty() ? std::string("-") : v.pci);
}

std::string attach_text(const Attach& a) {
    if (a.kind.empty()) return "unstated";
    std::string out = a.kind;
    if (a.pcie_gen) out += " gen" + std::to_string(a.pcie_gen);
    if (a.pcie_width) out += " x" + std::to_string(a.pcie_width);
    return out;
}

std::string roles_text(const Descriptor& d) {
    std::string out;
    for (const auto& role : d.roles) out += (out.empty() ? "" : ",") + role;
    return out.empty() ? "-" : out;
}

std::string rate_text(double value) {
    char text[32];
    std::snprintf(text, sizeof text, "%g", value);
    return text;
}

std::vector<Descriptor> other_units(const Declaration& declaration) {
    std::vector<Descriptor> out;
    for (const auto& d : declaration.devices)
        if (d.kind != Kind::gpu && !d.excluded) out.push_back(d);
    return out;
}

[[noreturn]] void refuse(const std::string& why) { throw Refusal("device gate: refusing to start: " + why); }

// Identity comparison of one declared entry against one enumerated device.
enum class Match { none, yes, conflict };
Match match(const Descriptor& d, const Visible& v) {
    // The enumerators list GPUs only. An entry of another kind that carries this GPU's address is a wrong file.
    if (d.kind == Kind::cpu) return Match::none;
    const bool uuid_known = !d.uuid.empty() && !v.uuid.empty();
    const bool pci_known = !d.pci.empty() && !v.pci.empty();
    const bool uuid_hit = uuid_known && d.uuid == v.uuid;
    const bool pci_hit = pci_known && d.pci == v.pci;
    if (!uuid_hit && !pci_hit) return Match::none;
    // One identity agrees and the other does not (a card moved slots, or a copied entry): never guess.
    if ((uuid_known && !uuid_hit) || (pci_known && !pci_hit)) return Match::conflict;
    if (d.vendor != v.vendor) return Match::conflict;
    return Match::yes;
}

std::string policy_denial(const Declaration& declaration, const Visible& v, const Descriptor* declared) {
    const auto name = lower(v.name);
    const auto declared_name = declared ? lower(declared->name) : std::string();
    for (const auto& p : declaration.policies) {
        if (!p.vendor.empty() && p.vendor != vendor_name(v.vendor)) continue;
        if (name.find(p.name_contains) == std::string::npos && declared_name.find(p.name_contains) == std::string::npos) continue;
        return "denied by policy " + p.name + (p.reason.empty() ? std::string() : " (" + p.reason + ")");
    }
    return {};
}

Descriptor implicit_integrated(const Visible& v) {
    Descriptor d;
    d.id = "integrated";
    d.vendor = v.vendor;
    d.arch = v.arch;
    d.uuid = v.uuid;
    d.pci = v.pci;
    d.name = v.name;
    d.memory_bytes = v.memory_bytes;
    d.attach.kind = "integrated";
    d.roles = {"pair", "decode", "prefill"};
    return d;
}

std::string json_string(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char buffer[8];
            std::snprintf(buffer, sizeof buffer, "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
            out += buffer;
        } else {
            out.push_back(c);
        }
    }
    return out + "\"";
}
} // namespace

std::string uuid_from_bytes(const unsigned char* bytes16) {
    std::string digits;
    char buffer[4];
    bool zero = true;
    for (int i = 0; i < 16; ++i) {
        std::snprintf(buffer, sizeof buffer, "%02x", static_cast<unsigned>(bytes16[i]));
        digits += buffer;
        zero = zero && bytes16[i] == 0;
    }
    return zero ? std::string() : normalise_uuid(digits);
}

std::string pci_from_parts(unsigned domain, unsigned bus, unsigned device, unsigned function) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%04x:%02x:%02x.%x", domain & 0xffffU, bus & 0xffU, device & 0xffU, function & 0x7U);
    return buffer;
}

Verdict classify(const Declaration& declaration, const Visible& visible) {
    Verdict out;
    std::size_t hits = 0;
    bool conflict = false;
    for (const auto& d : declaration.devices) {
        const auto m = match(d, visible);
        if (m == Match::none) continue;
        ++hits;
        conflict = conflict || m == Match::conflict;
        out.declared = &d;
    }
    // Deny wins, in this order: policy, ambiguity, explicit exclusion, then the conditions for doing work.
    if (auto denied = policy_denial(declaration, visible, hits == 1 ? out.declared : nullptr); !denied.empty()) {
        out.reason = std::move(denied);
        return out;
    }
    if (hits == 0) {
        out.reason = "undeclared";
        return out;
    }
    if (hits > 1 || conflict) {
        out.reason = hits > 1 ? "ambiguous identity: matches more than one declared device"
                              : "ambiguous identity: uuid, pci or vendor disagree with declared device " + out.declared->id;
        return out;
    }
    const auto& d = *out.declared;
    if (d.kind != Kind::gpu) out.reason = "declared as " + d.id + " with kind " + std::string(kind_name(d.kind)) + ", the device is a GPU";
    else if (d.excluded) out.reason = "excluded" + (d.reason.empty() ? std::string() : " (" + d.reason + ")");
    else if (d.roles.empty()) out.reason = "declared as " + d.id + " without a role";
    else if (!d.arch.empty() && !visible.arch.empty() && d.arch != lower(visible.arch))
        out.reason = "declared as " + d.id + " with arch " + d.arch + ", the device reports " + visible.arch;
    else out.eligible = true;
    return out;
}

Bound resolve(const Declaration* declaration, const std::vector<Visible>& all, const Request& request) {
    const std::string vendor(vendor_name(request.vendor));
    std::vector<Visible> visible;
    for (const auto& v : all)
        if (v.vendor == request.vendor) visible.push_back(v);
    if (visible.empty()) refuse("no " + vendor + " device is visible");

    if (!declaration) {
        if (request.vendor == Vendor::amd && visible.size() == 1 && visible.front().integrated && request.select.empty())
            return Bound{implicit_integrated(visible.front()), visible.front(), "implicit", {}};
        std::string list;
        for (const auto& v : visible) list += (list.empty() ? "" : "; ") + describe(v);
        refuse("no device declaration was found, so every " + vendor +
               " device is undeclared and excluded. Declare the device to use in a strix-devices-v1 file and pass --devices FILE "
               "or set STRIX_DEVICES. Visible: " + list);
    }

    std::vector<Verdict> verdicts;
    for (const auto& v : visible) verdicts.push_back(classify(*declaration, v));
    // Two enumerated devices claiming one declared entry: neither is trusted.
    for (std::size_t i = 0; i < visible.size(); ++i)
        for (std::size_t j = i + 1; j < visible.size(); ++j)
            if (verdicts[i].declared && verdicts[i].declared == verdicts[j].declared) {
                for (const auto k : {i, j}) {
                    if (verdicts[k].eligible) verdicts[k].reason = "ambiguous identity: two devices match declared device " + verdicts[k].declared->id;
                    verdicts[k].eligible = false;
                }
            }
    const auto listing = [&] {
        std::string out;
        for (std::size_t i = 0; i < visible.size(); ++i)
            out += (out.empty() ? "" : "; ") + describe(visible[i]) + ": " +
                   (verdicts[i].eligible ? "eligible as " + verdicts[i].declared->id : verdicts[i].reason);
        return " Visible: " + out + ". Declaration: " + declaration->source;
    };
    const auto bind = [&](std::size_t i) {
        if (!verdicts[i].eligible) refuse("device " + describe(visible[i]) + " is " + verdicts[i].reason + "." + listing());
        Bound bound{*verdicts[i].declared, visible[i], declaration->source, other_units(*declaration)};
        if (!request.role.empty())
            if (auto why = role_refusal(bound.device, request.role); !why.empty()) refuse(why + "." + listing());
        return bound;
    };

    if (!request.select.empty()) {
        const Descriptor* wanted = nullptr;
        for (const auto& d : declaration->devices)
            if (d.id == request.select) wanted = &d;
        if (!wanted) refuse("device id " + request.select + " is not declared." + listing());
        if (wanted->kind != Kind::gpu)
            refuse("device id " + request.select + " is a " + std::string(kind_name(wanted->kind)) + " unit, not a GPU." + listing());
        if (wanted->vendor != request.vendor) refuse("device id " + request.select + " is not a " + vendor + " device." + listing());
        for (std::size_t i = 0; i < visible.size(); ++i)
            if (verdicts[i].declared == wanted) return bind(i);
        refuse("declared device " + request.select + " is not visible." + listing());
    }
    if (request.ordinal >= 0) {
        for (std::size_t i = 0; i < visible.size(); ++i)
            if (visible[i].ordinal == request.ordinal) return bind(i);
        refuse("no " + vendor + " device at index " + std::to_string(request.ordinal) + "." + listing());
    }
    std::vector<std::size_t> eligible;
    for (std::size_t i = 0; i < visible.size(); ++i)
        if (verdicts[i].eligible) eligible.push_back(i);
    if (eligible.empty()) refuse("no visible " + vendor + " device is declared, included and given a role." + listing());
    if (eligible.size() > 1) {
        std::string ids;
        for (const auto i : eligible) ids += (ids.empty() ? "" : ", ") + verdicts[i].declared->id;
        refuse(std::to_string(eligible.size()) + " eligible " + vendor + " devices (" + ids +
               ") and none was selected: pass --device-id ID or set STRIX_DEVICE." + listing());
    }
    return bind(eligible.front());
}

std::string default_devices_path() {
#ifdef _WIN32
    const auto base = env("APPDATA");
    return base.empty() ? std::string() : base + "\\strix\\devices.toml";
#else
    if (const auto xdg = env("XDG_CONFIG_HOME"); !xdg.empty()) return xdg + "/strix/devices.toml";
    const auto home = env("HOME");
    return home.empty() ? std::string() : home + "/.config/strix/devices.toml";
#endif
}

Environment read_environment() {
    Environment out;
    out.devices_path = env("STRIX_DEVICES");
    out.select = env("STRIX_DEVICE");
    if (out.devices_path.empty()) {
        std::error_code ignored;
        if (const auto fallback = default_devices_path(); !fallback.empty() && std::filesystem::exists(fallback, ignored))
            out.devices_path = fallback;
    }
    return out;
}

Bound start_gate(Enumerator& enumerator, Request request, const std::string& devices_path) {
    const auto environment = read_environment();
    const auto path = devices_path.empty() ? environment.devices_path : devices_path;
    if (request.select.empty()) request.select = environment.select;
    request.vendor = enumerator.vendor();
    Declaration declaration;
    if (!path.empty()) {
        try {
            declaration = load_declaration(path);
        } catch (const std::exception& e) {
            refuse(std::string("the device declaration is not usable: ") + e.what());
        }
    }
    return resolve(path.empty() ? nullptr : &declaration, enumerator.enumerate(), request);
}

std::string units_text(const std::vector<Descriptor>& units) {
    std::string out;
    for (const auto& d : units) {
        out += (out.empty() ? "" : ",") + std::string(kind_name(d.kind)) + ":" + d.id + "[";
        std::string roles;
        for (const auto& role : d.roles) roles += (roles.empty() ? "" : ",") + role + (role_refusal(d, role).empty() ? "" : "(refused)");
        out += roles.empty() ? "-" : roles;
        if (!d.cores_text.empty()) out += " cores=" + d.cores_text;
        if (d.byte_cap_gbytes_ps > 0.0) out += " byte_cap=" + rate_text(d.byte_cap_gbytes_ps) + "GB/s";
        if (!d.corun_qualification.empty()) out += " qualified=" + d.corun_qualification;
        out += "]";
    }
    return out.empty() ? "-" : out;
}

std::string receipt_line(const Bound& bound) {
    const auto& d = bound.device;
    const auto& v = bound.visible;
    return "device: id=" + d.id + " vendor=" + std::string(vendor_name(v.vendor)) + " uuid=" + shown_uuid(v.vendor, v.uuid) +
           " pci=" + (v.pci.empty() ? std::string("-") : v.pci) + " name=\"" + v.name + "\" arch=" + (v.arch.empty() ? std::string("-") : v.arch) +
           " ordinal=" + std::to_string(v.ordinal) + " attach=" + attach_text(d.attach) + " roles=" + roles_text(d) +
           " declared=" + bound.source + (bound.units.empty() ? std::string() : " units=" + units_text(bound.units));
}

namespace {
// Absent without units, so a receipt from a GPU-only declaration keeps its bytes.
std::string units_json(const std::vector<Descriptor>& units) {
    if (units.empty()) return {};
    std::string out = ",\"units\":[";
    for (std::size_t i = 0; i < units.size(); ++i) {
        const auto& d = units[i];
        std::string granted, refused;
        for (const auto& role : d.roles) {
            auto& list = role_refusal(d, role).empty() ? granted : refused;
            list += (list.empty() ? "" : ",") + role;
        }
        out += std::string(i ? "," : "") + "{\"id\":" + json_string(d.id) + ",\"kind\":" + json_string(kind_name(d.kind)) +
               ",\"roles\":" + json_string(granted) + ",\"roles_refused\":" + json_string(refused);
        if (d.kind == Kind::cpu)
            out += ",\"cores\":" + json_string(d.cores_text) + ",\"byte_cap_gbytes_ps\":" + rate_text(d.byte_cap_gbytes_ps);
        if (d.kind == Kind::npu)
            out += ",\"pci\":" + json_string(d.pci) + ",\"driver\":" + json_string(d.driver) + ",\"firmware\":" + json_string(d.firmware) +
                   ",\"corun_qualification\":" + json_string(d.corun_qualification);
        out += "}";
    }
    return out + "]";
}
} // namespace

std::string receipt_json(const Bound& bound) {
    const auto& d = bound.device;
    const auto& v = bound.visible;
    return "{\"id\":" + json_string(d.id) + ",\"vendor\":" + json_string(vendor_name(v.vendor)) +
           ",\"uuid\":" + json_string(v.uuid.empty() ? std::string() : shown_uuid(v.vendor, v.uuid)) + ",\"pci\":" + json_string(v.pci) +
           ",\"name\":" + json_string(v.name) + ",\"arch\":" + json_string(v.arch) + ",\"attach\":" + json_string(attach_text(d.attach)) +
           ",\"roles\":" + json_string(roles_text(d)) + ",\"declared\":" + json_string(bound.source) + units_json(bound.units) + "}";
}

CommandLine take_command_line(int& argc, char** argv) {
    CommandLine out;
    int kept = 1;
    for (int i = 1; i < argc; ++i) {
        std::string* target = nullptr;
        if (std::strcmp(argv[i], "--devices") == 0) target = &out.devices_path;
        else if (std::strcmp(argv[i], "--device-id") == 0) target = &out.select;
        if (!target) {
            argv[kept++] = argv[i];
            continue;
        }
        if (i + 1 >= argc) throw std::invalid_argument(std::string(argv[i]) + " needs a value");
        *target = argv[++i];
    }
    for (int i = kept; i < argc; ++i) argv[i] = nullptr;
    argc = kept;
    return out;
}

} // namespace strix::device
