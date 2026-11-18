"""Compile the production host control flow with fault-injected CUDA calls, without CUDA headers."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
driver = (root / 'src/cuda/models/qwen/qwen_prefill_driver.cu').read_text()
start = driver.index('    const std::uint32_t valid_before = S.mtp_pooled_valid;')
body = driver.index('    for (std::uint32_t k = k0; k < (dexec ? k0 : k1);', start)
catch = driver.index('    } catch (...) {', body)
end = driver.index('    if (!per_step_sync) {', catch)
qsa = (root / 'src/cuda/models/qwen/qwen_qsa.cu').read_text()
shape = qsa[qsa.index('bool graph_shape('):qsa.index('\nvoid layer_core_device(')]
Path(sys.argv[1]).write_text('''
void transaction(bool capture, bool replay) {
    const unsigned depth = 2, pending = 7;
    const int dexec = replay ? 1 : 0;
    const unsigned k0 = 0, k1 = depth;  // one draft segment (the second segment reuses the same capture path)
''' + driver[start:body] + '''
    ck(step(), "device work");
    if (capture) {
        cudaGraph_t graph = nullptr;
        ck(cudaStreamEndCapture(m.stream, &graph), "end");
        cudaGraphDestroy(graph);
    }
''' + driver[catch:end] + '\n}\n' + shape, encoding='utf-8')
