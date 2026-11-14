#include "strix/device/gate.hpp"
#include "strix/device/hip_gate.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {
using namespace strix::device;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template <class F> void rejects(F f, const char* needle, const char* message) {
    try {
        f();
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(needle) == std::string::npos)
            throw std::runtime_error(std::string(message) + ": wrong error: " + e.what());
        return;
    }
    throw std::runtime_error(std::string(message) + ": accepted");
}
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

const char* const uuid_a = "GPU-11111111-2222-3333-4444-555555555555";
const char* const uuid_b = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
const char* const uuid_c = "GPU-99999999-8888-7777-6666-555555555555";

const char* const lab = R"(schema = "strix-devices-v1"
machine = "workstation"

[devices.main]
vendor = "nvidia"
uuid = "GPU-11111111-2222-3333-4444-555555555555"
arch = "sm_120"
memory_bytes = 102641958912
roles = ["replica", "prefill"]
attach = { kind = "pcie", pcie_gen = 4, pcie_width = 16, h2d_gbytes_ps = 0.0 }

[devices.spare]
vendor = "nvidia"
uuid = "GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
excluded = true
reason = "never schedule on this card"

[device_policies.no-spare]
name_contains = "4090"
excluded = true
reason = "lab rule"
)";

Visible nvidia(int ordinal, const char* name, const char* uuid, const char* pci, const char* arch = "sm_120") {
    Visible v;
    v.ordinal = ordinal;
    v.vendor = Vendor::nvidia;
    v.name = name;
    v.uuid = normalise_uuid(uuid);
    v.pci = normalise_pci(pci);
    v.arch = arch;
    return v;
}
Visible amd(int ordinal, const char* pci, bool integrated, const char* arch = "gfx1151") {
    Visible v;
    v.ordinal = ordinal;
    v.vendor = Vendor::amd;
    v.name = "AMD Radeon Graphics";
    v.pci = normalise_pci(pci);
    v.arch = arch;
    v.integrated = integrated;
    return v;
}
Request want(Vendor vendor, std::string select = {}, int ordinal = -1, std::string role = {}) {
    Request r;
    r.vendor = vendor;
    r.select = std::move(select);
    r.ordinal = ordinal;
    r.role = std::move(role);
    return r;
}

void identities() {
    require(normalise_uuid(uuid_b) == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "uuid normalise");
    require(normalise_uuid("AAAAAAAABBBBCCCCDDDDEEEEEEEEEEEE") == normalise_uuid(uuid_b), "uuid without dashes");
    rejects([] { (void)normalise_uuid("GPU-1234"); }, "32 hex", "short uuid");
    rejects([] { (void)normalise_uuid("00000000-0000-0000-0000-000000000000"); }, "all zero", "zero uuid");
    require(normalise_pci("C5:00.0") == "0000:c5:00.0", "pci short form");
    require(normalise_pci("00000000:01:00.0") == "0000:01:00.0", "pci long domain");
    rejects([] { (void)normalise_pci("1"); }, "pci", "pci index");
    rejects([] { (void)normalise_pci("0000:01:00.9"); }, "pci", "pci function");
    const unsigned char bytes[16] = {0x11, 0x11, 0x11, 0x11, 0x22, 0x22, 0x33, 0x33, 0x44, 0x44, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55};
    require(uuid_from_bytes(bytes) == normalise_uuid(uuid_a), "uuid bytes");
    const unsigned char zero[16] = {};
    require(uuid_from_bytes(zero).empty(), "zero uuid bytes are no identity");
    require(pci_from_parts(0, 0xc5, 0) == "0000:c5:00.0", "pci parts");
}

void listing() {
    const auto cards = parse_nvidia_smi("0, GPU-11111111-2222-3333-4444-555555555555, 00000000:01:00.0, 97887, NVIDIA RTX PRO 6000\r\n"
                                        "1, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 00000000:05:00.0, 24564, NVIDIA GeForce RTX 4090\n\n");
    require(cards.size() == 2 && cards[0].ordinal == 0 && cards[1].ordinal == 1, "listing size and order");
    require(cards[0].uuid == normalise_uuid(uuid_a) && cards[0].pci == "0000:01:00.0" && cards[0].name == "NVIDIA RTX PRO 6000", "listing identity");
    require(cards[1].memory_bytes == (24564ULL << 20U) && cards[1].name == "NVIDIA GeForce RTX 4090" && cards[1].vendor == Vendor::nvidia,
            "listing memory and name");
    require(parse_nvidia_smi("").empty(), "empty listing");
    rejects([] { (void)parse_nvidia_smi("No devices were found\n"); }, "nvidia-smi", "listing without fields");
    rejects([] { (void)parse_nvidia_smi("0, not-a-uuid, 00000000:01:00.0, 1, card\n"); }, "nvidia-smi", "listing with a bad uuid");
    rejects([] { (void)parse_nvidia_smi("x, GPU-11111111-2222-3333-4444-555555555555, 00000000:01:00.0, 1, card\n"); }, "nvidia-smi",
            "listing with a bad index");
    // The listing feeds the same gate: the excluded card is refused whatever its index.
    const auto d = parse_declaration(lab, "devices.toml");
    Request request;
    request.vendor = Vendor::nvidia;
    require(resolve(&d, cards, request).visible.uuid == normalise_uuid(uuid_a), "listing resolves to the declared card");
    request.select = "spare";
    rejects([&] { (void)resolve(&d, cards, request); }, "denied by policy", "listing cannot select the excluded card");
}

void parser() {
    const auto d = parse_declaration(lab);
    require(d.devices.size() == 2 && d.policies.size() == 1 && d.machine == "workstation", "declaration shape");
    require(d.devices[0].id == "main" && d.devices[0].roles.size() == 2 && d.devices[0].attach.pcie_width == 16, "main device");
    require(d.devices[1].excluded && d.devices[1].reason == "never schedule on this card", "spare device");
    const std::string head = "schema = \"strix-devices-v1\"\n";
    const std::string one = head + "[devices.a]\nvendor = \"amd\"\npci = \"c5:00.0\"\n";
    rejects([&] { (void)parse_declaration("[devices.a]\nvendor = \"amd\"\npci = \"c5:00.0\"\n"); }, "schema", "missing schema");
    rejects([&] { (void)parse_declaration("schema = \"strix-devices-v2\"\n"); }, "schema", "other schema");
    rejects([&] { (void)parse_declaration(head + "color = 1\n"); }, "unknown key", "unknown root key");
    rejects([&] { (void)parse_declaration(one + "index = 0\n"); }, "unknown key", "unknown device key");
    rejects([&] { (void)parse_declaration(head + "[devices.a]\nvendor = \"amd\"\n"); }, "identity", "device without identity");
    rejects([&] { (void)parse_declaration(one + "[devices.a]\nvendor = \"amd\"\npci = \"c6:00.0\"\n"); }, "", "duplicate id");
    rejects([&] { (void)parse_declaration(one + "[devices.b]\nvendor = \"amd\"\npci = \"0000:C5:00.0\"\n"); }, "same identity", "duplicate identity");
    rejects([&] { (void)parse_declaration(one + "roles = [\"render\"]\n"); }, "unknown role", "unknown role");
    rejects([&] { (void)parse_declaration(head + "[devices.a]\nvendor = \"arm\"\npci = \"c5:00.0\"\n"); }, "vendor", "unknown vendor");
    rejects([&] { (void)parse_declaration(one + "attach = { kind = \"usb4\" }\n"); }, "attach.kind", "unknown attach kind");
    rejects([&] { (void)parse_declaration(head + "[device_policies.p]\nname_contains = \"x\"\nexcluded = false\n"); }, "only exclude",
            "including policy");
    rejects([&] { (void)parse_declaration(head + "[device_policies.p]\nexcluded = true\n"); }, "name_contains", "policy without a pattern");

    // Roles that need a measured rate refuse until a probe fills it.
    const auto& main_device = d.devices[0];
    require(role_refusal(main_device, "replica").empty(), "replica needs no rate");
    require(role_refusal(main_device, "prefill").find("h2d_gbytes_ps") != std::string::npos, "prefill on an unprobed link refuses");
    require(role_refusal(main_device, "draft").find("not declared") != std::string::npos, "undeclared role refuses");
    auto probed = main_device;
    probed.attach.h2d_gbytes_ps = 6.0;
    require(role_refusal(probed, "prefill").empty(), "probed link allows prefill");
    auto remote = main_device;
    remote.attach = Attach{};
    remote.attach.kind = "remote";
    require(role_refusal(remote, "prefill").find("gbps_measured") != std::string::npos, "remote prefill needs the link rate");
    remote.attach = Attach{};
    require(role_refusal(remote, "prefill").find("no attach") != std::string::npos, "prefill without attach refuses");
    remote.attach.kind = "integrated";
    require(role_refusal(remote, "prefill").empty(), "integrated needs no rate");
}

void gate() {
    const auto d = parse_declaration(lab, "devices.toml");
    const auto main_card = nvidia(0, "NVIDIA RTX PRO 6000", uuid_a, "01:00.0");
    const auto spare_card = nvidia(1, "NVIDIA GeForce RTX 4090", uuid_b, "02:00.0");

    // Declared, included, with a role: bound, and the receipt names it.
    auto bound = resolve(&d, {main_card, spare_card}, want(Vendor::nvidia));
    require(bound.device.id == "main" && bound.visible.ordinal == 0, "single eligible device is chosen");
    const auto line = receipt_line(bound);
    require(line.find("device: id=main vendor=nvidia uuid=GPU-11111111-2222-3333-4444-555555555555 pci=0000:01:00.0") == 0, "receipt identity");
    require(line.find("attach=pcie gen4 x16 roles=replica,prefill declared=devices.toml") != std::string::npos, "receipt attach and roles");
    require(receipt_json(bound).find("\"id\":\"main\"") != std::string::npos, "receipt json");

    // Index reorder with the same UUIDs selects the same card.
    auto swapped_main = main_card;
    auto swapped_spare = spare_card;
    swapped_main.ordinal = 1;
    swapped_spare.ordinal = 0;
    bound = resolve(&d, {swapped_spare, swapped_main}, want(Vendor::nvidia));
    require(bound.device.id == "main" && bound.visible.ordinal == 1 && bound.visible.uuid == main_card.uuid, "reorder keeps the card");

    // Declared and excluded refuses, by id, by index, and when it is the only card.
    rejects([&] { (void)resolve(&d, {main_card, spare_card}, want(Vendor::nvidia, "spare")); }, "denied by policy", "excluded by id");
    rejects([&] { (void)resolve(&d, {main_card, spare_card}, want(Vendor::nvidia, {}, 1)); }, "refusing to start", "excluded by index");
    rejects([&] { (void)resolve(&d, {spare_card}, want(Vendor::nvidia)); }, "no visible nvidia device is declared", "only the excluded card");
    auto no_policy = d;
    no_policy.policies.clear();
    rejects([&] { (void)resolve(&no_policy, {spare_card}, want(Vendor::nvidia, "spare")); }, "excluded (never schedule on this card)",
            "excluded flag alone");

    // Undeclared refuses; a new UUID after a card swap is still caught by the policy line.
    const auto stranger = nvidia(0, "NVIDIA RTX A5000", uuid_c, "03:00.0");
    rejects([&] { (void)resolve(&d, {stranger}, want(Vendor::nvidia)); }, "undeclared", "undeclared device");
    rejects([&] { (void)resolve(&d, {stranger}, want(Vendor::nvidia, {}, 0)); }, "is undeclared", "undeclared device by index");
    const auto replaced_spare = nvidia(0, "NVIDIA GeForce RTX 4090", uuid_c, "02:00.0");
    rejects([&] { (void)resolve(&d, {replaced_spare}, want(Vendor::nvidia, {}, 0)); }, "denied by policy no-spare (lab rule)", "policy on a new uuid");

    // Policy deny beats a role.
    auto roled = d;
    roled.devices[1].excluded = false;
    roled.devices[1].roles = {"replica"};
    rejects([&] { (void)resolve(&roled, {spare_card}, want(Vendor::nvidia, "spare")); }, "denied by policy", "policy beats a role");
    require(!classify(roled, spare_card).eligible && classify(roled, main_card).eligible, "classification");

    // Two eligible devices and no selection refuses; an id or a verified index selects.
    roled.policies.clear();
    rejects([&] { (void)resolve(&roled, {main_card, spare_card}, want(Vendor::nvidia)); }, "none was selected", "two devices, no selection");
    require(resolve(&roled, {main_card, spare_card}, want(Vendor::nvidia, "spare")).visible.ordinal == 1, "selection by id");
    require(resolve(&roled, {main_card, spare_card}, want(Vendor::nvidia, {}, 0)).device.id == "main", "selection by verified index");
    rejects([&] { (void)resolve(&roled, {main_card}, want(Vendor::nvidia, "spare")); }, "is not visible", "selected device absent");
    rejects([&] { (void)resolve(&roled, {main_card}, want(Vendor::nvidia, "other")); }, "is not declared", "unknown id");

    // Ambiguous identity excludes: same uuid on two devices, or uuid and pci pointing at different entries.
    auto twin = main_card;
    twin.ordinal = 1;
    twin.pci = normalise_pci("04:00.0");
    rejects([&] { (void)resolve(&d, {main_card, twin}, want(Vendor::nvidia)); }, "ambiguous identity", "two devices, one entry");
    auto pinned = d;
    pinned.devices[0].pci = normalise_pci("01:00.0");
    auto moved = main_card;
    moved.pci = normalise_pci("05:00.0");
    rejects([&] { (void)resolve(&pinned, {moved}, want(Vendor::nvidia)); }, "ambiguous identity", "uuid and pci disagree");
    require(resolve(&pinned, {main_card}, want(Vendor::nvidia)).device.id == "main", "uuid and pci agree");

    // No role, wrong arch, wrong role, no device, no declaration.
    auto idle = d;
    idle.devices[0].roles.clear();
    rejects([&] { (void)resolve(&idle, {main_card}, want(Vendor::nvidia)); }, "without a role", "declared without a role");
    const auto old_arch = nvidia(0, "NVIDIA RTX PRO 6000", uuid_a, "01:00.0", "sm_89");
    rejects([&] { (void)resolve(&d, {old_arch}, want(Vendor::nvidia)); }, "arch", "arch mismatch");
    rejects([&] { (void)resolve(&d, {main_card}, want(Vendor::nvidia, {}, -1, "draft")); }, "role draft", "role not declared");
    require(resolve(&d, {main_card}, want(Vendor::nvidia, {}, -1, "replica")).device.id == "main", "declared role");
    rejects([&] { (void)resolve(&d, {}, want(Vendor::nvidia)); }, "no nvidia device is visible", "nothing visible");
    rejects([&] { (void)resolve(nullptr, {main_card}, want(Vendor::nvidia)); }, "no device declaration", "nvidia without a declaration");
    rejects([&] { (void)resolve(&d, {amd(0, "c5:00.0", true)}, want(Vendor::nvidia)); }, "no nvidia device is visible", "other vendor only");

    // AMD: a lone integrated device keeps working without a file; anything else needs the declaration.
    const auto igpu = amd(0, "c5:00.0", true);
    const auto dgpu = amd(1, "03:00.0", false, "gfx1100");
    bound = resolve(nullptr, {igpu}, want(Vendor::amd));
    require(bound.source == "implicit" && bound.device.attach.kind == "integrated" && bound.visible.pci == "0000:c5:00.0", "implicit integrated");
    require(receipt_line(bound).find("declared=implicit") != std::string::npos, "implicit receipt");
    rejects([&] { (void)resolve(nullptr, {igpu, dgpu}, want(Vendor::amd)); }, "no device declaration", "two HIP devices without a file");
    rejects([&] { (void)resolve(nullptr, {dgpu}, want(Vendor::amd)); }, "no device declaration", "discrete HIP device without a file");
    const auto node = parse_declaration("schema = \"strix-devices-v1\"\n[devices.igpu]\nvendor = \"amd\"\narch = \"gfx1151\"\npci = \"0000:c5:00.0\"\n"
                                        "attach = { kind = \"integrated\" }\nroles = [\"pair\", \"decode\", \"prefill\"]\n");
    require(resolve(&node, {dgpu, igpu}, want(Vendor::amd)).device.id == "igpu", "PCI selection with an undeclared second device");
    rejects([&] { (void)resolve(&node, {dgpu}, want(Vendor::amd)); }, "undeclared", "undeclared second HIP device");
    require(resolve(&node, {igpu}, want(Vendor::amd, {}, -1, "prefill")).device.id == "igpu", "integrated prefill needs no rate");
}

void entry_points() {
    const auto path = (std::filesystem::temp_directory_path() / "strix-device-gate-tests.toml").string();
    {
        std::ofstream out(path, std::ios::binary);
        out << lab;
    }
    FakeEnumerator cards(Vendor::nvidia, {nvidia(0, "NVIDIA GeForce RTX 4090", uuid_b, "02:00.0"), nvidia(1, "NVIDIA RTX PRO 6000", uuid_a, "01:00.0")});
    set_env("STRIX_DEVICE", "");
    set_env("STRIX_DEVICES", path.c_str());
    require(start_gate(cards, {}).visible.ordinal == 1, "STRIX_DEVICES is read");
    set_env("STRIX_DEVICE", "spare");
    rejects([&] { (void)start_gate(cards, {}); }, "denied by policy", "STRIX_DEVICE cannot select an excluded device");
    set_env("STRIX_DEVICE", "");
    set_env("STRIX_DEVICES", (path + ".missing").c_str());
    rejects([&] { (void)start_gate(cards, {}); }, "not usable", "missing declaration file refuses");
    require(start_gate(cards, {}, path).device.id == "main", "explicit path beats the environment");
    set_env("STRIX_DEVICES", "");
    std::filesystem::remove(path);

    char a0[] = "app", a1[] = "--model", a2[] = "m.gguf", a3[] = "--devices", a4[] = "d.toml", a5[] = "--device-id", a6[] = "main", a7[] = "--ctx";
    char* argv[] = {a0, a1, a2, a3, a4, a5, a6, a7, nullptr};
    int argc = 8;
    const auto cli = take_command_line(argc, argv);
    require(cli.devices_path == "d.toml" && cli.select == "main" && argc == 4, "flags are taken");
    require(std::string(argv[1]) == "--model" && std::string(argv[3]) == "--ctx" && argv[4] == nullptr, "other arguments keep their order");
    char b0[] = "app", b1[] = "--devices";
    char* bad_argv[] = {b0, b1, nullptr};
    int bad_argc = 2;
    rejects([&] { (void)take_command_line(bad_argc, bad_argv); }, "needs a value", "flag without a value");
}

// NPU and CPU units: declared explicitly, never matched against the GPU listing, shown in the start line.
void units() {
    const std::string gpu = std::string("schema = \"strix-devices-v1\"\n[devices.main]\nvendor = \"amd\"\npci = \"0000:c5:00.0\"\n"
                                        "roles = [\"pair\", \"decode\", \"prefill\", \"aux\"]\nattach = { kind = \"integrated\" }\n");
    const std::string npu = "[devices.npu]\nkind = \"npu\"\nvendor = \"amd\"\npci = \"0000:c5:00.1\"\narch = \"aie2p\"\nroles = [\"aux\"]\n";
    const std::string qualified = "driver = \"amdxdna\"\nfirmware = \"1.1.2.64\"\ncorun_qualification = \"E1-test\"\n";
    const std::string host = "[devices.host]\nkind = \"cpu\"\ncores = \"2-15\"\nroles = [\"host\", \"aux\"]\nbyte_cap_gbytes_ps = 10.0\n";

    const auto all = parse_declaration(gpu + npu + qualified + host);
    require(all.devices.size() == 3, "three units parse");
    const Descriptor* cpu = host_unit(all);
    require(cpu && cpu->id == "host" && cpu->kind == Kind::cpu, "host unit found");
    require(cpu->cores.size() == 14 && cpu->cores.front() == 2 && cpu->cores.back() == 15 && cpu->cores_text == "2-15", "cores parsed");
    require(cpu->byte_cap_gbytes_ps == 10.0, "byte cap parsed");
    require(units_for_role(all, "aux").size() == 3 && units_for_role(all, "host").size() == 1 && units_for_role(all, "decode").size() == 1,
            "units by role");
    require(parse_cores("0,4-7,12") == std::vector<std::uint32_t>({0, 4, 5, 6, 7, 12}), "core list");
    for (const char* bad : {"", "a", "3-1", "1,1", "0-2,2", "1,", "-3", "4096", "1 2"})
        rejects([&] { (void)parse_cores(bad); }, "cores", "bad core list");

    // An old file is unchanged: kind defaults to gpu, and the receipt carries no units text.
    const auto old = parse_declaration(gpu);
    require(old.devices.front().kind == Kind::gpu && host_unit(old) == nullptr, "default kind is gpu, no implicit host");
    Visible igpu;
    igpu.ordinal = 0;
    igpu.vendor = Vendor::amd;
    igpu.name = "integrated";
    igpu.pci = "0000:c5:00.0";
    igpu.integrated = true;
    const auto plain = resolve(&old, {igpu}, Request{});
    require(plain.units.empty() && receipt_line(plain).find("units=") == std::string::npos &&
                receipt_json(plain).find("units") == std::string::npos,
            "gpu-only receipt unchanged");
    require(resolve(nullptr, {igpu}, Request{}).units.empty(), "the implicit device brings no npu or cpu");

    const auto bound = resolve(&all, {igpu}, Request{});
    require(bound.device.id == "main" && bound.units.size() == 2, "gpu bound, two other units listed");
    const auto line = receipt_line(bound);
    require(line.find(" units=cpu:host[host,aux cores=2-15 byte_cap=10GB/s],npu:npu[aux qualified=E1-test]") != std::string::npos,
            "units in the start line");
    const auto json = receipt_json(bound);
    require(json.find("\"units\":[{\"id\":\"host\",\"kind\":\"cpu\",\"roles\":\"host,aux\",\"roles_refused\":\"\",\"cores\":\"2-15\","
                      "\"byte_cap_gbytes_ps\":10}") != std::string::npos &&
                json.find("\"corun_qualification\":\"E1-test\"}]}") != std::string::npos,
            "units in the receipt");
    rejects([&] { (void)resolve(&all, {igpu}, Request{Vendor::amd, "npu", -1, {}}); }, "npu unit, not a GPU", "selecting the npu as the GPU");
    rejects([&] { (void)resolve(&all, {igpu}, Request{Vendor::amd, "host", -1, {}}); }, "cpu unit, not a GPU", "selecting the cpu as the GPU");

    // An NPU without its qualification parses, is listed, and is refused the role.
    const auto unqualified = parse_declaration(gpu + npu + "driver = \"amdxdna\"\n");
    const Descriptor& bare = unqualified.devices.back();
    require(bare.kind == Kind::npu && role_refusal(bare, "aux").find("needs firmware") != std::string::npos, "npu aux needs firmware");
    require(units_for_role(unqualified, "aux").size() == 1, "an unqualified npu takes no aux work");
    require(units_text(resolve(&unqualified, {igpu}, Request{}).units) == "npu:npu[aux(refused)]", "refused role is printed");
    require(role_refusal(parse_declaration(gpu + npu).devices.back(), "aux").find("needs driver") != std::string::npos, "npu aux needs driver");
    require(role_refusal(parse_declaration(gpu + npu + "driver = \"d\"\nfirmware = \"f\"\n").devices.back(), "aux").find("corun_qualification") !=
                std::string::npos,
            "npu aux needs the co-run qualification");
    // An excluded unit is not listed and gets no work.
    const auto off = parse_declaration(gpu + host + "excluded = true\n");
    require(host_unit(off) == nullptr && resolve(&off, {igpu}, Request{}).units.empty(), "excluded cpu unit");

    // A GPU whose address is declared as an NPU is a wrong file, not an undeclared card.
    const auto swapped = parse_declaration(std::string("schema = \"strix-devices-v1\"\n[devices.npu]\nkind = \"npu\"\nvendor = \"amd\"\n"
                                                       "pci = \"0000:c5:00.0\"\nroles = [\"aux\"]\n"));
    rejects([&] { (void)resolve(&swapped, {igpu}, Request{}); }, "with kind npu, the device is a GPU", "gpu address declared as npu");

    const auto bad = [&](const std::string& text, const char* needle, const char* message) {
        rejects([&] { (void)parse_declaration(text); }, needle, message);
    };
    bad(gpu + "kind = \"tpu\"\n", "expected gpu, npu or cpu", "unknown kind");
    bad(gpu + npu.substr(0, npu.find("roles")) + "roles = [\"aux\", \"decode\"]\n", "not available to a npu unit", "decode on an npu");
    bad(gpu + npu.substr(0, npu.find("roles")) + "roles = [\"host\"]\n", "not available to a npu unit", "host on an npu");
    bad(gpu + "[devices.host]\nkind = \"cpu\"\nroles = [\"prefill\"]\n", "not available to a cpu unit", "prefill on the cpu");
    bad(std::string("schema = \"strix-devices-v1\"\n[devices.main]\nvendor = \"amd\"\npci = \"0000:c5:00.0\"\nroles = [\"host\"]\n"),
        "not available to a gpu unit", "host on a gpu");
    bad(gpu + "cores = \"2-15\"\n", "only a kind = \"cpu\" unit", "cores on a gpu");
    bad(gpu + "byte_cap_gbytes_ps = 10.0\n", "only a kind = \"cpu\" unit", "byte cap on a gpu");
    bad(gpu + "driver = \"x\"\n", "only a kind = \"npu\" unit", "driver on a gpu");
    bad(gpu + host + "pci = \"0000:00:00.0\"\n", "a cpu unit takes no pci", "identity on the cpu");
    bad(gpu + host + "[devices.host2]\nkind = \"cpu\"\nroles = [\"host\"]\n", "second cpu unit", "two cpu units");
    bad(gpu + "[devices.npu]\nkind = \"npu\"\nvendor = \"amd\"\nroles = [\"aux\"]\n", "needs an identity", "npu without identity");
    bad(gpu + "[devices.npu]\nkind = \"npu\"\npci = \"0000:c5:00.1\"\n", "vendor: required", "npu without vendor");
    bad(gpu + "[devices.host]\nkind = \"cpu\"\ncores = \"15-2\"\n", "reversed range", "reversed cores");
    bad(gpu + "[devices.host]\nkind = \"cpu\"\nbyte_cap_gbytes_ps = 0\n", "must be positive", "zero byte cap");
}

// The host wait policy text (hip_gate.hpp is free of HIP headers): engine choice, override, refusal.
void host_wait() {
    using strix::device::HostWait;
    using strix::device::resolve_host_wait;
    require(resolve_host_wait(HostWait::blocking, nullptr) == HostWait::blocking, "host wait: engine choice without override");
    require(resolve_host_wait(HostWait::spin, "") == HostWait::spin, "host wait: empty override");
    require(resolve_host_wait(HostWait::spin, "block") == HostWait::blocking, "host wait: block override");
    require(resolve_host_wait(HostWait::blocking, "spin") == HostWait::spin, "host wait: spin override");
    require(resolve_host_wait(HostWait::blocking, "yield") == HostWait::yield, "host wait: yield override");
    require(resolve_host_wait(HostWait::blocking, "runtime") == HostWait::runtime, "host wait: runtime override");
    bool refused = false;
    try {
        (void)resolve_host_wait(HostWait::spin, "1");
    } catch (const strix::device::Refusal&) {
        refused = true;
    }
    require(refused, "host wait: unknown text is refused");
}
} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        identities();
        listing();
        parser();
        gate();
        entry_points();
        units();
        host_wait();
    } catch (const std::exception& e) {
        std::cerr << "device gate tests: " << e.what() << '\n';
        return 1;
    }
    std::cout << "device gate tests: ok\n";
    return 0;
}
