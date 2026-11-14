#pragma once

#include "strix/device/declaration.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The start gate: one function every GPU binary calls before its first allocation.
//
// A device does work only if it is declared, not excluded and has a role. Undeclared = excluded. Deny wins: an
// `excluded` entry or a matching policy beats any role, and an ambiguous identity excludes. The enumeration index is
// never an identity: a card is matched by UUID and/or PCI address, so a reorder cannot move work to another card.
namespace strix::device {

// One device as the vendor runtime reports it. Enumeration reads properties only; it creates no context.
struct Visible {
    int ordinal = -1; // runtime index, valid for this process only
    Vendor vendor = Vendor::amd;
    std::string name;
    std::string uuid; // normalised, may be empty
    std::string pci;  // normalised, may be empty
    std::string arch; // "sm_120", "gfx1151"
    std::uint64_t memory_bytes = 0;
    bool integrated = false;
};

// Identity text from what the vendor runtimes report (16 raw UUID bytes; PCI domain/bus/device, function 0).
[[nodiscard]] std::string uuid_from_bytes(const unsigned char* bytes16);
[[nodiscard]] std::string pci_from_parts(unsigned domain, unsigned bus, unsigned device, unsigned function = 0);

class Enumerator {
public:
    virtual ~Enumerator() = default;
    [[nodiscard]] virtual Vendor vendor() const noexcept = 0;
    [[nodiscard]] virtual std::vector<Visible> enumerate() = 0;
};

// NVIDIA cards as `nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.total,name --format=csv,noheader,nounits`
// prints them. This listing does not start the CUDA runtime, so the process can still be restricted to one card
// afterwards. `ordinal` is the nvidia-smi index. parse throws std::invalid_argument; query returns nullopt when the
// tool is missing, fails or prints something unexpected.
[[nodiscard]] std::vector<Visible> parse_nvidia_smi(std::string_view csv);
[[nodiscard]] std::optional<std::vector<Visible>> query_nvidia_smi();
// Sets a variable in this process's environment (used to restrict the vendor runtime before it starts). Throws Refusal.
void set_process_environment(const char* name, const std::string& value);

class FakeEnumerator final : public Enumerator {
public:
    FakeEnumerator(Vendor vendor, std::vector<Visible> devices) : vendor_(vendor), devices_(std::move(devices)) {}
    [[nodiscard]] Vendor vendor() const noexcept override { return vendor_; }
    [[nodiscard]] std::vector<Visible> enumerate() override { return devices_; }

private:
    Vendor vendor_;
    std::vector<Visible> devices_;
};

struct Request {
    Vendor vendor = Vendor::amd;
    std::string select; // declared device id; empty = the single eligible device
    int ordinal = -1;   // an operator-given runtime index: it must name an eligible device, it is never an identity
    std::string role;   // optional: the device must carry this role
};

struct Bound {
    Descriptor device;
    Visible visible;
    std::string source; // declaration path, or "implicit" for the undeclared single integrated device
    // The declared NPU and CPU units (not excluded), for the lanes that give them work. Empty without a declaration:
    // a unit that is not a GPU is never implicit.
    std::vector<Descriptor> units;
};

// Thrown for every refusal; what() is the one line to show the operator.
struct Refusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Why one enumerated device may or may not do work. `declared` is null for an undeclared device.
struct Verdict {
    const Descriptor* declared = nullptr;
    bool eligible = false;
    std::string reason; // empty when eligible
};
[[nodiscard]] Verdict classify(const Declaration& declaration, const Visible& visible);

// Pure decision. `declaration` is null when no file was found: every device is then undeclared, except that a lone
// integrated AMD device keeps working (the behaviour before declarations existed).
[[nodiscard]] Bound resolve(const Declaration* declaration, const std::vector<Visible>& visible, const Request& request);

// Where the declaration and the selection come from, outside the command line.
struct Environment {
    std::string devices_path; // STRIX_DEVICES, else the per-user default file when it exists, else empty
    std::string select;       // STRIX_DEVICE
};
[[nodiscard]] Environment read_environment();
[[nodiscard]] std::string default_devices_path(); // may not exist; empty when no home directory is known

// Loads the declaration (explicit path > STRIX_DEVICES > per-user default), enumerates and resolves.
// `request.select` empty falls back to STRIX_DEVICE. Throws Refusal.
[[nodiscard]] Bound start_gate(Enumerator& enumerator, Request request, const std::string& devices_path = {});

// "device: id=main vendor=nvidia uuid=GPU-... pci=0000:01:00.0 name="..." arch=sm_120 ordinal=0 attach=pcie gen4 x16
//  roles=replica,prefill declared=/path/devices.toml"
// With declared NPU or CPU units the line continues: ` units=npu:npu[aux],cpu:host[host,aux cores=2-15 byte_cap=10GB/s]`
// (a role the unit may not take yet is printed as `aux(refused)`); the JSON form gains a "units" array.
[[nodiscard]] std::string receipt_line(const Bound& bound);
// "npu:npu[aux],cpu:host[host,aux cores=2-15 byte_cap=10GB/s]"; "-" when there is none.
[[nodiscard]] std::string units_text(const std::vector<Descriptor>& units);
// {"id":"main","vendor":"nvidia","uuid":"GPU-...","pci":"...","name":"...","attach":"pcie gen4 x16",...}
[[nodiscard]] std::string receipt_json(const Bound& bound);

// Removes `--devices FILE` and `--device-id ID` from an argument vector (in place) and returns them, so every binary
// accepts the two flags without touching its own parser. Throws std::invalid_argument for a flag without a value.
struct CommandLine {
    std::string devices_path;
    std::string select;
};
[[nodiscard]] CommandLine take_command_line(int& argc, char** argv);

} // namespace strix::device
