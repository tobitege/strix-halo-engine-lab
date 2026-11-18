#!/usr/bin/env bash
# Post-install smoke test of the Strix Linux packages (installed system-wide or extracted into a directory).
#   smoke.sh [--root DIR] [--prefix /usr] [--port 18080] [--model FILE.gguf] [--require-health]
# --root: where the packages were extracted (default /, i.e. installed); binaries are <root><prefix>/bin.
# Checks: every installed strix-* executable loads and answers --help (no missing library, no crash);
# strix-skeleton-serve answers GET /health = 200 without weights; with --model, strix-hip-decode-token
# --print-effective-config resolves the model. Prints one PASS/FAIL/SKIP line per check; exit 1 on any FAIL.
set -u
# Shared CI entry point: listing errors must fail before the forbidden-content search.
if [ "${1:-}" = "--check-packages" ]; then
    [ $# -eq 2 ] || exit 2
    listing=$(mktemp) || exit 1
    trap 'rm -f "$listing"' EXIT
    count=0
    for package in "$2"/strix-*.deb "$2"/strix-*.rpm "$2"/strix-*.tar.gz "$2"/strix-*.zip; do
        [ -f "$package" ] || continue
        count=$((count + 1))
        case "$package" in
            *.deb) dpkg-deb -c "$package" >>"$listing" || exit 1 ;;
            *.rpm) rpm -qlp "$package" >>"$listing" || exit 1 ;;
            *.tar.gz) tar -tzf "$package" >>"$listing" || exit 1 ;;
            *.zip) unzip -Z1 "$package" >>"$listing" || exit 1 ;;
        esac
    done
    [ "$count" -gt 0 ] || { echo 'FAIL no packages to inspect'; exit 1; }
    grep -E '(^|/)(lab|artifacts)/|\.gguf$|\.safetensors$|mmproj' "$listing"
    result=$?
    [ "$result" -eq 1 ] || { echo 'FAIL forbidden content or listing search error'; exit 1; }
    echo "PASS inspected $count packages"
    exit 0
fi
root=/ prefix=/usr port=18080 model="" require_health=0
while [ $# -gt 0 ]; do
    case "$1" in
        --root) root="$2"; shift 2 ;;
        --prefix) prefix="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --require-health) require_health=1; shift ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "smoke.sh: unknown option $1" >&2; exit 2 ;;
    esac
done
base="${root%/}${prefix}"
bindir="$base/bin" datadir="$base/share/strix"
fails=0 passes=0 nohelp=0 nogpu=0
pass() { passes=$((passes + 1)); echo "PASS $*"; }
fail() { fails=$((fails + 1)); echo "FAIL $*"; }
skip() { echo "SKIP $*"; }

[ -d "$bindir" ] || { echo "FAIL no bin directory at $bindir"; exit 1; }
expected=0 checked=0
for manifest in "$datadir"/support/executables-*.txt; do
    [ -f "$manifest" ] || continue
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        expected=$((expected + 1))
        [ -x "$bindir/$name" ] || fail "expected executable missing: $name"
    done <"$manifest"
done
[ "$expected" -gt 0 ] || fail 'no expected executable manifest installed'
for exe in "$bindir"/strix-*; do
    [ -x "$exe" ] || continue
    checked=$((checked + 1))
    name=$(basename "$exe")
    if file -b "$exe" 2>/dev/null | grep -q ELF && command -v ldd >/dev/null 2>&1; then
        missing=$(ldd "$exe" 2>/dev/null | grep "not found" | head -3 | tr -s ' \t' ' ')
        if [ -n "$missing" ]; then fail "$name: missing libraries:$missing"; continue; fi
    fi
    without_help=0
    for manifest in "$datadir"/support/without-help-*.txt; do
        [ -f "$manifest" ] || continue
        if grep -Fxq "$name" "$manifest"; then without_help=1; break; fi
    done
    if [ "$without_help" -eq 1 ]; then
        out=$(timeout 30 "$exe" 2>&1 </dev/null); rc=$?
        if { [ "$rc" -eq 1 ] || [ "$rc" -eq 2 ]; } && [ -n "$out" ] &&
           ! printf '%s' "$out" | grep -qiE 'error while loading shared libraries|symbol lookup error|Traceback|ImportError|ModuleNotFoundError'; then
            nohelp=$((nohelp + 1)); pass "$name without --help (rc=$rc)"
        else
            fail "$name without arguments: rc=$rc: $(printf '%s' "$out" | head -2 | tr '\n' ' ' | cut -c1-160)"
        fi
        continue
    fi
    out=$(timeout 30 "$exe" --help 2>&1 </dev/null); rc=$?
    if [ $rc -eq 124 ] || [ $rc -ge 128 ]; then
        fail "$name --help: rc=$rc (timeout or signal)"
    elif [ $rc -eq 0 ]; then
        pass "$name --help (rc=$rc)"
    elif { [ "${out%%$'\n'*}" = "usage: $name" ] ||
           [[ "$out" == "usage: $name "* ]] ||
           [[ "$out" == "$name: usage: $name "* ]]; } &&
         ! printf '%s' "$out" | grep -qiE 'Traceback|ImportError|ModuleNotFoundError'; then
        pass "$name usage (rc=$rc)"
    elif printf '%s' "$out" | grep -qiE "error while loading shared libraries|symbol lookup error"; then
        fail "$name --help: rc=$rc: $(printf '%s' "$out" | head -2 | tr '\n' ' ' | cut -c1-160)"
    elif { [ $rc -eq 1 ] || [ $rc -eq 2 ]; } && printf '%s' "$out" | grep -qiE "device gate: refusing to start: cannot enumerate"; then
        # Libraries resolved and the start gate ran; with no visible GPU nothing more can run (HIP mains bind first).
        nogpu=$((nogpu + 1))
        pass "$name loads (no visible GPU: the device gate refused before --help)"
    else
        # Only these known no-help responses establish that the program loaded.
        parser_ok=0
        if [ $rc -eq 1 ]; then
            case "$name:$out" in
                'strix-memory-plan:error: unknown argument: --help') parser_ok=1 ;;
            esac
            if [ "$name" = strix-deepseek41-expert-server ]; then
                case "$out" in
                    $'{"event":"POISONED","error":"unknown option --help"}\nerror: unknown option --help'|\
                    $'error: unknown option --help\n{"event":"POISONED","error":"unknown option --help"}') parser_ok=1 ;;
                esac
            fi
        fi
        if [ "$parser_ok" -eq 1 ]; then
            nohelp=$((nohelp + 1)); pass "$name known parser response (rc=$rc)"
        else
            fail "$name --help: rc=$rc: $(printf '%s' "$out" | head -2 | tr '\n' ' ' | cut -c1-160)"
        fi
    fi
done
[ "$checked" -gt 0 ] || fail 'no executables checked'

probe() {
    if command -v curl >/dev/null 2>&1; then
        curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$port/health"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import sys,urllib.request as u
try: print(u.urlopen('http://127.0.0.1:$port/health', timeout=2).status)
except Exception: print(0)"
    else
        echo 0
    fi
}
skel="$bindir/strix-skeleton-serve" reg="$datadir/examples/skeleton_engine"
if [ -x "$skel" ] && [ -f "$reg/models/registry.toml" ]; then
    log=$(mktemp)
    "$skel" --registry-root "$reg" --registry models/registry.toml --model skeleton --bind 127.0.0.1 \
        --port "$port" >"$log" 2>&1 </dev/null &
    pid=$!
    code=0
    for _ in $(seq 1 50); do
        code=$(probe)
        [ "$code" = "200" ] && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
    done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    if [ "$code" = "200" ]; then pass "strix-skeleton-serve GET /health = 200"
    else fail "strix-skeleton-serve GET /health = $code: $(tail -n 3 "$log" | tr '\n' ' ' | cut -c1-200)"; fi
    rm -f "$log"
elif [ $require_health -eq 1 ]; then
    fail "health: strix-skeleton-serve or its registry not installed (install strix-core)"
else
    skip "health: strix-skeleton-serve not in this package (strix-core has it)"
fi

hip="$bindir/strix-hip-decode-token"
if [ -x "$hip" ] && [ -n "$model" ]; then
    out=$(timeout 120 "$hip" "$model" 1 --http-serve --print-effective-config 2>&1 </dev/null); rc=$?
    if [ $rc -eq 0 ]; then pass "strix-hip-decode-token --print-effective-config ($(printf '%s' "$out" | wc -l) lines)"
    else
        why=$(printf '%s' "$out" | grep -v '^strix-route-receipt' | grep -m 2 -iE "error|refus|invalid|required|cannot|missing")
        [ -n "$why" ] || why=$(printf '%s' "$out" | grep -v '^strix-route-receipt' | tail -n 2)
        fail "strix-hip-decode-token --print-effective-config rc=$rc: $(printf '%s' "$why" | tr '\n' ' ' | cut -c1-240)"
    fi
elif [ -x "$hip" ]; then
    skip "--print-effective-config: needs --model FILE.gguf"
fi
echo "smoke: $passes passed ($nohelp without --help, $nogpu stopped by the device gate without a GPU), $fails failed"
[ $fails -eq 0 ]
