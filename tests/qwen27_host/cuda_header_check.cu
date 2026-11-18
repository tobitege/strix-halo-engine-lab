// Compile-only integration check; contains no kernels and never opens a device.
#include "strix/qwen27_model.hpp"
#include "model_graph/dense_adapter.hpp"
static_assert(strix::weights::Nvfp4::block_size == 16);
void qwen27_cuda_host_contract_check(const strix::qwen27::Model& model) {
    (void)model.config.hidden;
}
