#include <exception>
#include <iostream>
#include <string_view>
#if defined(STRIX_WFMT_CASE_CPU)
bool wfmt_cpu_test();
bool wfmt_cpu_capture(std::string_view);
#else
bool wfmt_intel_test();
bool wfmt_intel_capture(std::string_view);
#endif
int main(int argc,char** argv) try {
 bool capture=argc==3 && std::string_view(argv[1])=="--wfmt-backend-capture";
#if defined(STRIX_WFMT_CASE_CPU)
 const bool ok=capture?wfmt_cpu_capture(argv[2]):wfmt_cpu_test();
#else
 const bool ok=capture?wfmt_intel_capture(argv[2]):wfmt_intel_test();
#endif
 std::cout << (ok?"PASS\n":"FAIL\n"); return ok?0:1;
} catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
