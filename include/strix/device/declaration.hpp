#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// strix-devices-v1: the device declaration every GPU binary reads before its first allocation.
//
//   schema = "strix-devices-v1"
//   machine = "workstation"                      # optional, display only
//
//   [devices.main]                               # the table key is the device id
//   vendor = "nvidia"                            # amd | nvidia | intel
//   uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"   # identity: uuid and/or pci, never an index
//   arch = "sm_120"                              # optional, checked when present
//   memory_bytes = 102641958912                  # optional, informational
//   roles = ["replica", "prefill"]               # empty or absent = no work
//   attach = { kind = "pcie", pcie_gen = 4, pcie_width = 16, h2d_gbytes_ps = 0.0 }
//
//   [devices.spare]
//   vendor = "nvidia"
//   uuid = "GPU-..."
//   excluded = true
//   reason = "never schedule on this card"
//
//   [device_policies.no-spare]                   # deny rules matched on the enumerated name
//   name_contains = "4090"
//   excluded = true
//   reason = "never schedule on this card"
//
//   [devices.npu]                                # a unit that is not a GPU states its kind
//   kind = "npu"                                 # gpu (default) | npu | cpu
//   vendor = "amd"
//   pci = "0000:c5:00.1"                         # identity as for a GPU
//   arch = "aie2p"
//   roles = ["aux"]                              # side models only (score, vision, embed)
//   driver = "amdxdna"                           # the three keys below are required before `aux` is granted
//   firmware = "1.1.2.64"
//   corun_qualification = "E1-2026-11"           # id of the GPU + NPU co-run receipt
//
//   [devices.host]
//   kind = "cpu"                                 # no identity; at most one cpu unit
//   cores = "2-15"                               # cores the host lane may pin
//   roles = ["host", "aux"]                      # host: tokenizer, image, sampling, exchange fold, background work
//   byte_cap_gbytes_ps = 10.0                    # background memory traffic ceiling while any decode is active
//
// Unknown keys, a device without an identity, duplicate identities and unknown roles are errors. An NPU or CPU that
// is not declared gets no work: there is no implicit non-GPU unit. `decode`, `prefill`, `pair`, `draft`, `replica`,
// `front` and `experts` are GPU roles; an NPU takes only `aux`; a CPU takes only `host` and `aux`.
namespace strix::device {

enum class Vendor { amd, nvidia, intel };
[[nodiscard]] std::string_view vendor_name(Vendor vendor) noexcept;
[[nodiscard]] Vendor parse_vendor(std::string_view text); // throws std::invalid_argument

enum class Kind { gpu, npu, cpu };
[[nodiscard]] std::string_view kind_name(Kind kind) noexcept;
[[nodiscard]] Kind parse_kind(std::string_view text); // throws std::invalid_argument

struct Attach {
    std::string kind;            // "" (not stated) | integrated | pcie | oculink | remote
    std::uint32_t pcie_gen = 0;  // negotiated, not the capability
    std::uint32_t pcie_width = 0;
    double h2d_gbytes_ps = 0.0;  // 0 = not probed
    double gbps_measured = 0.0;  // remote only; 0 = not probed
    std::uint64_t rtt_us_measured = 0;
};

struct Descriptor {
    std::string id;
    Kind kind = Kind::gpu;
    Vendor vendor = Vendor::amd; // optional for a cpu unit
    std::string arch;
    std::string uuid; // normalised: lower-case, "gpu-" prefix removed; empty when not declared
    std::string pci;  // normalised: dddd:bb:dd.f lower-case; empty when not declared
    std::string name; // display only
    std::uint64_t memory_bytes = 0;
    Attach attach;
    std::vector<std::string> roles;
    bool excluded = false;
    std::string reason;
    // kind = cpu only.
    std::string cores_text;           // as declared ("2-15", "0,4-7"); empty = not declared
    std::vector<std::uint32_t> cores; // sorted, unique
    double byte_cap_gbytes_ps = 0.0;  // 0 = not declared
    // kind = npu only: what was qualified. `aux` is refused while any of the three is missing.
    std::string driver;
    std::string firmware;
    std::string corun_qualification;
};

struct Policy {
    std::string name;
    std::string name_contains; // case-insensitive substring of the enumerated device name
    std::string vendor;        // optional: restricts the rule to one vendor
    bool excluded = true;
    std::string reason;
};

struct Declaration {
    std::string source; // path or "<inline>"
    std::string machine;
    std::vector<Descriptor> devices;
    std::vector<Policy> policies;
};

// Throws std::invalid_argument naming the offending key.
[[nodiscard]] Declaration parse_declaration(std::string_view toml, std::string source = "<inline>");
// Additionally throws std::runtime_error when the file cannot be read.
[[nodiscard]] Declaration load_declaration(const std::string& path);

// Identity normalisation shared by the parser and the enumerators. Both throw std::invalid_argument.
[[nodiscard]] std::string normalise_uuid(std::string_view text); // 32 hex digits -> 8-4-4-4-12
[[nodiscard]] std::string normalise_pci(std::string_view text);  // [dddd:]bb:dd.f -> dddd:bb:dd.f

// "2-15", "0,4-7,12" -> sorted unique core indices. Throws std::invalid_argument (empty, reversed range, repeat, > 4095).
[[nodiscard]] std::vector<std::uint32_t> parse_cores(std::string_view text);

[[nodiscard]] bool known_role(std::string_view role) noexcept;
// Whether a unit of this kind may declare the role at all (the parser refuses the file otherwise).
[[nodiscard]] bool role_allowed(Kind kind, std::string_view role) noexcept;
// Empty when the descriptor may take the role; otherwise the reason it may not (role absent, or a role that needs a
// measured rate on a link that was never probed, or `aux` on an NPU without driver, firmware and co-run qualification).
[[nodiscard]] std::string role_refusal(const Descriptor& device, std::string_view role);

// The declared units that may take `role` now (not excluded, no refusal), in declaration order. GPU entries are
// included: whether a declared GPU is present is the start gate's question, not this one's.
[[nodiscard]] std::vector<const Descriptor*> units_for_role(const Declaration& declaration, std::string_view role);
// The cpu unit when it is declared, not excluded and carries `host`; otherwise null (no host lane work).
[[nodiscard]] const Descriptor* host_unit(const Declaration& declaration) noexcept;

} // namespace strix::device
