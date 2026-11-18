#pragma once

#include "strix/device/link_class.hpp"

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
//   profile = "rtx-5090-32"                      # optional known model (profiles.hpp): every route sizes itself to it
//   memory_gib = 30                              # optional device budget (over the profile's memory)
//
//   [devices.pair]                               # a planned shape: no identity, planners only
//   profile = "rtx-5090-32"
//   count = 2
//   interconnect = "pcie5x16"
//
//   [devices.halo-a]                             # a GPU in another machine, reached over the network
//   vendor = "amd"
//   pci = "0000:c3:00.0"
//   roles = ["prefill", "decode"]
//   attach = { kind = "remote", link = "2.5gbe" }        # link class: the one line that sets the link model
//   # attach = { kind = "remote", link = "2.5gbe", port = "nic0", gbps_measured = 2.31, rtt_us_measured = 180 }
//
//   [links.pair]                                 # a direct link between two declared devices (not via this host)
//   between = ["halo-a", "halo-b"]
//   class = "usb4"                               # same classes as attach.link; measured keys override the class
//   gbps_measured = 36.8                         # optional; rtt_us_measured and port as in attach
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
//   memory_gib = 64                              # optional host RAM budget for the routes on this machine
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
    double gbps_measured = 0.0;  // remote only: useful payload Gbit/s; 0 = not probed
    std::uint64_t rtt_us_measured = 0;
    // Link class (device/link_class.hpp: 1gbe, 2.5gbe, 5gbe, 10gbe, 25gbe, usb4, usb4-gen2, pcie3x4, pcie4x4, pcie4x16,
    // pcie5x16, oculink4x4). Nominal rates only; a measured key above always wins. Empty = derive from kind/pcie keys.
    std::string link;
    // Shared resource name: links that name the same port share its bandwidth. Default: "attach-remote" for a remote
    // device (every remote device of one machine shares its network port unless declared otherwise), else
    // "attach-<device id>" (a PCIe slot is its own resource).
    std::string port;
};

// A declared link between two devices that does not pass through this machine (for example the cable of a pair).
struct Link {
    std::string name;
    std::string a, b; // device ids of this declaration
    std::string link; // class, as Attach::link (required)
    double gbps_measured = 0.0;
    std::uint64_t rtt_us_measured = 0;
    std::string port; // shared resource; default "link-<name>"
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
    // Known device model (strix/device/profiles.hpp) and the entry's overrides of it. budget_bytes is `memory_gib`:
    // on a GPU the device budget every route sizes itself to (over the profile's memory), on the cpu unit the host
    // RAM budget. 0 = not declared.
    std::string profile;
    std::uint64_t budget_bytes = 0;
    double bandwidth_gbytes_ps = 0.0; // overrides the profile's memory bandwidth; 0 = the profile's
    std::uint32_t units = 0;          // overrides the profile's SM / CU count; 0 = the profile's
    // Cluster shape: `count` identical devices joined by `interconnect` (a class in profiles.hpp). An entry with
    // count > 1, or a profile without an identity, is planned: the planners size it, the gate never binds it.
    std::uint32_t count = 1;
    std::string interconnect;
    bool planned = false;
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
    std::vector<Link> links;
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
