#!/usr/bin/env python3
# decode-study-client.py PORT OUT.jsonl ARMS [MAXTOK] : experiment 0/1 client. ARMS like S8,C2,C4,C8,C8,C4,C2,S8
# Sn = lanes 0..n-1 served one after another; Cn = lanes 0..n-1 concurrently (multi-request cohort path).
# Every lane = shared prefix (spec) + its own sub-task. Greedy, thinking off, return_token_ids.
import json, sys, threading, time, urllib.request, hashlib
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import importlib.util, os
from pathlib import Path
prompt_path = Path(os.environ.get("TURBODEC_PROMPT_MODULE", str(Path(__file__).with_name("decode-study-classes.py"))))
_spec = importlib.util.spec_from_file_location("decode_study_prompts", prompt_path)
if _spec is None or _spec.loader is None: raise RuntimeError("cannot load prompt module: " + str(prompt_path))
_t = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_t)
PREFIX, LANES = _t.PREFIX, _t.LANES

port, out_path, arms = int(sys.argv[1]), sys.argv[2], sys.argv[3].split(',')
maxtok = int(sys.argv[4]) if len(sys.argv) > 4 else 256
for name in arms:
    if (len(name) < 2 or name[0] not in 'SC' or not name[1:].isdigit()
            or not 1 <= int(name[1:]) <= len(LANES)):
        raise SystemExit(f'invalid arm {name!r}: prompt module has {len(LANES)} lanes')
url = f"http://127.0.0.1:{port}/v1/chat/completions"

def body(i):
    return {"messages": [{"role": "system", "content": PREFIX}, {"role": "user", "content": LANES[i]}],
            "max_tokens": maxtok, "enable_thinking": False, "return_token_ids": True}

def one(i, res, t0):
    s = time.time()
    try:
        req = urllib.request.Request(url, data=json.dumps(body(i)).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.loads(r.read())
        e = time.time()
        st = j.get("strix", {})
        ids = st.get("token_ids", [])
        res[i] = {"lane": i, "start": s - t0, "end": e - t0, "tokens": j["usage"]["completion_tokens"],
                  "prompt": j["usage"]["prompt_tokens"], "cached": j["usage"].get("prompt_tokens_details", {}).get("cached_tokens"),
                  "queue_ms": st.get("queue_ms"), "prefill_ms": st.get("prefill_ms"), "decode_ms": st.get("decode_ms"),
                  "dec_tps": st.get("decode_tokens_per_second"), "acc": st.get("mtp_accepted"), "prop": st.get("mtp_proposed"),
                  "cycles": st.get("mtp_cycles"), "ids_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
                  "finish": j["choices"][0]["finish_reason"]}
    except Exception as e:
        res[i] = {"lane": i, "error": repr(e)}; return
    res[i]["request_sha256"] = hashlib.sha256(json.dumps(json.loads(req.data), sort_keys=True).encode()).hexdigest()
    if os.environ.get("TURBODEC_KEEP_TOKEN_IDS") == "1":
        res[i]["token_ids"] = ids

def arm(name):
    n = int(name[1:]); t0 = time.time(); res = [None] * n
    if name[0] == 'S':
        for i in range(n): one(i, res, t0)
    else:
        th = [threading.Thread(target=one, args=(i, res, t0)) for i in range(n)]
        for t in th: t.start()
        for t in th: t.join()
    wall = time.time() - t0
    ok = [r for r in res if r and "error" not in r]
    for r in ok:
        emitted = (int(r["acc"]) + int(r["cycles"])) if r.get("acc") is not None and r.get("cycles") is not None else None
        r["cycle_emitted_tokens"] = emitted
        r["prefill_or_noncycle_tokens"] = r["tokens"] - emitted if emitted is not None else None
        r["cycle_accounting_valid"] = r["prefill_or_noncycle_tokens"] == 1
    tok = sum(r["tokens"] for r in ok)
    cycle_valid = bool(ok) and all(r["cycle_accounting_valid"] for r in ok)
    cycle_tokens = sum(r["cycle_emitted_tokens"] for r in ok) if cycle_valid else None
    cycle_count = sum(r["cycles"] or 0 for r in ok)
    first = [r["start"] + (r["queue_ms"] + r["prefill_ms"]) / 1000 for r in ok]
    dwin = (max(r["end"] for r in ok) - min(first)) if ok else 0
    summary = {"arm": name, "t0": t0, "wall_s": round(wall, 3), "tokens": tok, "agg_wall_tps": round(tok / wall, 2),
               "agg_decode_window_tps": round(tok / dwin, 2) if dwin > 0 else None,
               "sum_lane_dec_tps": round(sum(r["dec_tps"] or 0 for r in ok), 2),
               "mean_lane_dec_tps": round(sum(r["dec_tps"] or 0 for r in ok) / max(1, len(ok)), 2),
               "acc_rate": round(sum(r["acc"] or 0 for r in ok) / max(1, sum(r["prop"] or 0 for r in ok)), 3),
               "cycle_emitted_tokens": cycle_tokens, "cycle_accounting_valid": cycle_valid,
               "cycle_tokens_per_cycle": round(cycle_tokens / max(1, cycle_count), 6) if cycle_tokens is not None else None,
               "legacy_client_tokens_per_cycle": round(tok / max(1, cycle_count), 6),
               "errors": [r for r in res if not r or "error" in r], "lanes": res}
    print(json.dumps({k: v for k, v in summary.items() if k != "lanes"}), flush=True)
    return summary

failed = False
with open(out_path, "x") as f:
    for a in arms:
        result = arm(a)
        f.write(json.dumps(result) + "\n"); f.flush()
        failed |= bool(result["errors"]) or any(r["tokens"] <= 0 for r in result["lanes"] if r and "error" not in r)
sys.exit(1 if failed else 0)
