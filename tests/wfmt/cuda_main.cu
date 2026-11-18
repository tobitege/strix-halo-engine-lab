#include "strix/device/cuda_enumerator.hpp"
#include "../../src/cuda/weight/registry.hpp"
#include <cstdio>
#include <exception>
#include <string_view>

namespace strix::weight::test {
bool cuda_dense_case();
#ifdef STRIX_WFMT_GSQ
bool cuda_gsq_case();
#endif
#ifdef STRIX_WFMT_NVFP4
bool cuda_nvfp4_case();
#endif
}
namespace {
bool abi_case() {
    namespace w = strix::weight;
    namespace cw = strix::cuda::weight;
    w::Plan plan{}; plan.selected.backend = w::Backend::cuda;
    plan.selected.storage = {"gguf.q8_0",1};
    plan.selected.kernel = cw::dense_project_schema;
    w::WeightView view{}; view.storage = plan.selected.storage;
    w::ExecutionContext context{}; context.backend = w::Backend::cuda;
    cw::DenseProjectArguments dense{};
    auto args = cw::arguments(dense);
    auto reject = [&](w::Error expected) {
        return cw::enqueue_registered(plan,view,args,&dense,nullptr,0,context).code == expected;
    };
    args.abi_revision = 2;
    if (!reject(w::Error::malformed)) return false;
    args = cw::arguments(dense); --args.bytes;
    if (!reject(w::Error::malformed)) return false;
    args = cw::arguments(dense); args.data = nullptr;
    if (!reject(w::Error::malformed)) return false;
    args = cw::arguments(dense); args.schema = 0xdeadbeef;
    if (!reject(w::Error::unsupported_arithmetic)) return false;
    args = cw::arguments(dense); context.backend = w::Backend::hip;
    if (!reject(w::Error::unsupported_backend)) return false;
    context.backend = w::Backend::cuda; view.storage = {"plain.f32",1};
    if (!reject(w::Error::unknown_format)) return false;
    view.storage = plan.selected.storage; plan.workspace_bytes = 256;
    return reject(w::Error::memory_budget);
}
}
int main(int argc, char** argv) try {
    if (argc == 2 && std::string_view(argv[1]) == "cuda-abi")
        return abi_case() ? 0 : 1; // validates rejection without starting CUDA
    strix::device::bind_cuda_main(argc,argv);
    if (argc != 2) return 2;
    const std::string_view name(argv[1]);
    bool pass = false;
    if (name == "cuda-dense") pass = strix::weight::test::cuda_dense_case();
#ifdef STRIX_WFMT_GSQ
    else if (name == "cuda-gsq") pass = strix::weight::test::cuda_gsq_case();
#endif
#ifdef STRIX_WFMT_NVFP4
    else if (name == "cuda-nvfp4") pass = strix::weight::test::cuda_nvfp4_case();
#endif
    else return 2;
    std::printf("%s %s\n",argv[1],pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
} catch (const std::exception& error) {
    std::fprintf(stderr,"wfmt CUDA: %s\n",error.what()); return 1;
}
