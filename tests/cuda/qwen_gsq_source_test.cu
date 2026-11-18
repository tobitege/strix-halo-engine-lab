// Device oracle for the signed GSQ operand sources: decoded codes, scales and ordered group products against the
// host reference for every format of the GSQ pack (OperandExact).
#include "../wfmt/cuda_gsq.cu"

int main() {
    const bool ok = strix::weight::test::cuda_gsq_case();
    std::printf("qwen_gsq_source_test %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
