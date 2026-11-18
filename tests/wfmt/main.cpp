#include "cases.hpp"
#include <iostream>
#include <exception>
#include <string_view>
int wfmt_census_test();
bool wfmt_readers_test();
bool wfmt_admission_test();
bool wfmt_contracts_test();
bool wfmt_reference_test();
bool wfmt_strides_test();
bool wfmt_arithmetic_test();
bool wfmt_group_reads_test();
bool wfmt_geometry_test();
bool wfmt_iq3s_test();
bool wfmt_reference_capture(std::string_view);
bool wfmt_resolve_test();
bool wfmt_cache_test();
bool wfmt_effective_config_test();
bool wfmt_runtime_test();
bool wfmt_cpu_test();
bool wfmt_intel_test();
bool wfmt_cpu_capture(std::string_view);
bool wfmt_intel_capture(std::string_view);
int main(int argc, char** argv) try {
    if (argc == 3 && std::string_view(argv[1]) == "--wfmt-backend-capture")
        return wfmt_cpu_capture(argv[2]) && wfmt_intel_capture(argv[2]) ? 0 : 1;
    if (argc == 3 && std::string_view(argv[1]) == "--wfmt-reference-capture")
        return wfmt_reference_capture(argv[2]) ? 0 : 1;
    if (argc == 5 && std::string_view(argv[1]) == "--wfmt-capture" &&
        std::string_view(argv[3]) == "--wfmt-matrix")
        return strix::weight::test::capture_case(argv[2]) ? 0 : 1;
    if (argc != 2) return 2;
    const std::string_view name(argv[1]);
    bool ok = false;
    if (name == "resolve") ok = wfmt_resolve_test();
    else if (name == "cpu") ok = wfmt_cpu_test();
    else if (name == "intel") ok = wfmt_intel_test();
    else if (name == "cache") ok = wfmt_cache_test();
    else if (name == "effective-config") ok = wfmt_effective_config_test();
    else if (name == "runtime") ok = wfmt_runtime_test();
    else
    if (name == "contracts") ok = wfmt_contracts_test();
    else if (name == "census") ok = wfmt_census_test() == 0;
    else if (name == "readers") ok = wfmt_readers_test();
    else if (name == "admission") ok = wfmt_admission_test();
    else if (name == "compare") ok = strix::weight::test::compare_case();
    else if (name == "reference") ok = wfmt_reference_test();
    else if (name == "strides") ok = wfmt_strides_test();
    else if (name == "arithmetic") ok = wfmt_arithmetic_test();
    else if (name == "group-reads") ok = wfmt_group_reads_test();
    else if (name == "geometry") ok = wfmt_geometry_test();
    else if (name == "iq3s") ok = wfmt_iq3s_test();
    else return 2;
    std::cout << name << (ok ? " PASS\n" : " FAIL\n");
    return ok ? 0 : 1;

}

catch(const std::exception& e) { std::cerr << "wfmt: " << e.what() << "\n"; return 1; }
