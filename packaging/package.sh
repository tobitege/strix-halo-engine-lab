#!/usr/bin/env bash
# Build and package one Linux release tree (docs/install.md).
#   packaging/package.sh BUILD_DIR host|hip|cuda-qwen [--hip-archs "gfx1151;gfx1201"] [--cuda-archs 120]
#                        [--rpath DIR] [--jobs N] [--generators "TGZ;DEB;RPM"] [--require-tag] [-- CMAKE_ARGS...]
# host: strix-core, strix-tools, strix-docs. hip: strix-hip (HIP targets from --hip-archs, default gfx1151; the
# package description and share/strix/support/build-hip.txt mark each one qualified or unqualified).
# cuda-qwen: strix-cuda (Linux CUDA, compiled only, not qualified). Only the packaged executables are built
# (CMAKE_SKIP_INSTALL_ALL_DEPENDENCY). Logs go to BUILD_DIR/*.log; the last lines list the packages.
# Wrap the whole command in the host's build-lane script when one is required (e.g. a shared build node).
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
[ $# -ge 2 ] || { sed -n '2,9p' "$0"; exit 2; }
build=$1 mode=$2; shift 2
hip_archs=gfx1151 cuda_archs=120 rpath="" jobs=$(nproc) generators="" require_tag=OFF extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --hip-archs) hip_archs=$2; shift 2 ;;
        --cuda-archs) cuda_archs=$2; shift 2 ;;
        --rpath) rpath=$2; shift 2 ;;
        --jobs) jobs=$2; shift 2 ;;
        --generators) generators=$2; shift 2 ;;
        --require-tag) require_tag=ON; shift ;;
        --) shift; extra=("$@"); break ;;
        *) echo "package.sh: unknown option $1" >&2; exit 2 ;;
    esac
done
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_TESTING=OFF -DSTRIX_PACKAGING=ON
    -DSTRIX_PACKAGE_REQUIRE_TAG=$require_tag -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON -DSTRIX_LEGACY_MAINS=OFF
    "-DSTRIX_PACKAGE_INSTALL_RPATH=$rpath")
[ -n "$generators" ] && common+=("-DCPACK_GENERATOR=$generators")
case "$mode" in
    host) opts=(-DSTRIX_ENABLE_HIP_RUNTIME=OFF) ;;
    hip) opts=(-DSTRIX_ENABLE_HIP_RUNTIME=ON "-DSTRIX_HIP_ARCHS=$hip_archs" "-DCMAKE_HIP_ARCHITECTURES=$hip_archs") ;;
    cuda-qwen) opts=(-DSTRIX_CUDA_QWEN_ONLY=ON "-DCMAKE_CUDA_ARCHITECTURES=$cuda_archs") ;;
    *) echo "package.sh: mode is host, hip or cuda-qwen" >&2; exit 2 ;;
esac
mkdir -p "$build"
cmake -S "$src" -B "$build" "${common[@]}" "${opts[@]}" "${extra[@]}" >"$build/configure.log" 2>&1 ||
    { tail -n 20 "$build/configure.log"; exit 1; }
flavor=$mode
receipt="$build/packaging/build-$flavor.txt"
# The executables of the components this tree packages, from the configure receipt ("  component:target").
case "$mode" in host) comps='core\|tools' ;; hip) comps='hip' ;; *) comps='cuda' ;; esac
mapfile -t targets < <(sed -n "s/^  \\($comps\\):\\(strix-[A-Za-z0-9_.-]*\\)\$/\\2/p" "$receipt")
[ ${#targets[@]} -gt 0 ] || { echo "package.sh: no packaged executables in $receipt" >&2; exit 1; }
cmake --build "$build" -j "$jobs" --target "${targets[@]}" >"$build/build.log" 2>&1 ||
    { grep -m 12 -E "error:|FAILED:" "$build/build.log" | cut -c1-300; tail -n 5 "$build/build.log"; exit 1; }
(cd "$build" && cpack >cpack.log 2>&1) || { tail -n 20 "$build/cpack.log"; exit 1; }
echo "package.sh: $mode built ${#targets[@]} executables"
ls -1 "$build"/strix-*.deb "$build"/strix-*.rpm "$build"/strix-*.tar.gz 2>/dev/null || true
