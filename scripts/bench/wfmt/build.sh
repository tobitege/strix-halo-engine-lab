#!/usr/bin/env bash
# Invoke through the deployment's CPU build lane. Never starts a GPU ticket.
set -euo pipefail
source_dir= build_dir= backend= config=Release toolchain=
targets=()
while (($#)); do
    case "$1" in
        --source) source_dir=$2; shift 2;;
        --build) build_dir=$2; shift 2;;
        --backend) backend=$2; shift 2;;
        --config) config=$2; shift 2;;
        --toolchain) toolchain=$2; shift 2;;
        --target) targets+=("$2"); shift 2;;
        *) echo "unknown argument: $1" >&2; exit 2;;
    esac
done
[[ -n $source_dir && -n $build_dir && -n $backend ]] || {
    echo 'required: --source --build --backend' >&2; exit 2;
}
[[ $config == Release || $config == RelWithDebInfo ]] || exit 2
mkdir -p "$build_dir"
build_dir=$(cd "$build_dir" && pwd)
flags=(-DBUILD_TESTING=ON -DSTRIX_WFMT_TESTS=ON -DCMAKE_BUILD_TYPE="$config")
[[ -z $toolchain ]] || flags+=(-DCMAKE_TOOLCHAIN_FILE="$toolchain")
cmake_source=$source_dir
target=strix-wfmt-host-tests
case "$backend" in
    host) cmake_source=$source_dir/tests/wfmt; flags+=(-DSTRIX_WFMT_BACKEND=host);;
    hip) flags+=(-DSTRIX_ENABLE_HIP_RUNTIME=ON -DSTRIX_HIP_ARCHS=gfx1151 -DSTRIX_FAMILY_MINIMAX=ON); target=strix-wfmt-hip-tests;;
    cuda-qwen) flags+=(-DSTRIX_CUDA_QWEN_ONLY=ON); target=strix-wfmt-cuda-tests;;
    cuda-ds41) flags+=(-DSTRIX_CUDA_DS41_ONLY=ON -DSTRIX_ENABLE_CUDA_DS41=ON -DSTRIX_ENABLE_CUDA_TRELLIS=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF); target=strix-wfmt-cuda-tests;;
    cpu) flags+=(-DSTRIX_BACKEND_CPU=ON); target=strix-wfmt-cpu-tests;;
    intel) cmake_source=$source_dir/src/intel; target=strix-wfmt-intel-tests;;
    nvfp4) cmake_source=$source_dir/tests/nvfp4; flags+=(-DSTRIX_NVFP4_CUDA=ON); target=nvfp4_host;;
    *) echo "unsupported backend: $backend" >&2; exit 2;;
esac
targets=("$target" "${targets[@]}")
status=0
cmake -S "$cmake_source" -B "$build_dir" "${flags[@]}" > "$build_dir/wfmt-configure.log" 2>&1 || status=$?
if ((status == 0)); then
    cmake --build "$build_dir" --config "$config" --target "${targets[@]}" > "$build_dir/wfmt-build.log" 2>&1 || status=$?
fi
printf 'schema=wfmt.build@1\nbackend=%s\nconfig=%s\nstatus=%s\n' "$backend" "$config" "$status" > "$build_dir/wfmt-summary.txt"
git -C "$source_dir" rev-parse HEAD >> "$build_dir/wfmt-summary.txt"
cmake --version | head -1 >> "$build_dir/wfmt-summary.txt"
cat "$build_dir/wfmt-summary.txt"
exit "$status"
