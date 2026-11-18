#!/usr/bin/env bash
# Vision content battery on the DeepSeek V4.1 pair server: one greedy answer per image, graded from the pixels.
#
#   scripts/bench/common/ticket.sh OUT vision/battery --bin-dir DIR --model MODEL --vision VISION.gguf --images IMGDIR
#       [--arm 'NAME|ARGS|ENV'] [--port 18763] [--prefill-rows 2048]
#   afterwards (CPU): python3 scripts/bench/vision/score.py IMGDIR OUT/*/battery.json
#
# IMGDIR: synthetic PNGs from tools/vision/make_images.py or tools/vision/ds41_ref.py (names carry the generator
# kind: grad, checker, palette, text, noise, rings). The rubric (score.py): colour classes present >= 5 % of the
# pixels must be named, classes below 0.5 % must not; pattern and orientation words; palette_blocks names red,
# green, blue and yellow. A fast probe for a broken image path: solid red / green / blue / yellow images.
# Rank 0 serves (with --vision), rank 1 runs on KIT_PEER; the server run carries the usual validity receipt.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bdir='' model='' vision='' images='' arm='base||' port=18763 rows=2048 pbdir='' pmodel=''
while (($#)); do
    case $1 in
        --bin-dir) bdir=$2 ;; --model) model=$2 ;; --vision) vision=$2 ;; --images) images=$2 ;; --arm) arm=$2 ;;
        --port) port=$2 ;; --prefill-rows) rows=$2 ;; --peer-bin-dir) pbdir=$2 ;; --peer-model) pmodel=$2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
bin=$bdir/strix-deepseek41-serve
[[ -x $bin && -e $model && -f $vision && -d $images ]] || kit_die 'need --bin-dir, --model, --vision and --images'
pbin=${pbdir:-$bdir}/strix-deepseek41-serve pmodel=${pmodel:-$model}
kit_same_binary "$bin" "$pbin"
IFS='|' read -r name args envs <<<"$arm"; read -ra A <<<"$args"; read -ra E <<<"$envs"
B=(STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000)
common=(--transport usb4auto --peer-timeout 120 --prefill-path device --slots 8 --prefill-rows "$rows"
    --http-port "$port" --http-bind 127.0.0.1)
peer=$(printf '%q ' env "${B[@]}" "${E[@]}" "$pbin" "$pmodel" --rank 1 "${common[@]}" "${A[@]}")
client=$(printf '%q ' "$KIT_PY" -I "$here/battery.py" "$port")' "$KIT_RUN_DIR/battery.json" '$(printf %q "$name")' '$(printf '%q ' "$images"/*.png)
bash "$here/../common/serve-decode.sh" --tag "$name" --arm "$name" --port "$port" --ready-log '"READY"' \
    --peer-cmd "$peer" --client-cmd "$client" -- env "${B[@]}" "${E[@]}" "$bin" "$model" --rank 0 "${common[@]}" \
    --vision "$vision" "${A[@]}"
[[ -s ${KIT_OUT:-bench-out}/$name/battery.json ]] && "$KIT_PY" -I "$here/score.py" "$images" "${KIT_OUT:-bench-out}/$name/battery.json" |
    grep -a '^TOTAL' || echo "== $name battery: no answers (see $name/client.txt)"
