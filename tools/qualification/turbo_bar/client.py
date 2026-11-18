"""OpenAI-compatible client for the task and retrieval items (urllib only).

  python -m tools.qualification.turbo_bar.client run --url URL --items tasks.jsonl --arm NAME --out DIR [--concurrency 8]
  python -m tools.qualification.turbo_bar.client score --items tasks.jsonl --results DIR [--out DIR]

``run`` POSTs {url}/v1/chat/completions (temperature 0, top_p 1, fixed seed, thinking disabled, non-streaming)
and writes DIR/results.NAME.jsonl: reply text, usage, latency and any "strix" receipt object. Plain items use a
pool of --concurrency threads. Items with a ``group`` (retrieval) run group by group: the optional warm-up
request first (sequential), then the group's items with pool size = the item's ``concurrency``.
``score`` writes DIR/scored.NAME.jsonl (one row per item with ok) using the tasks/retrieval scorers.
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.qualification.turbo_bar import tasks
from tools.qualification.turbo_bar import manifest as mf
from tools.qualification.turbo_bar.common import (ITEM_FIELDS, canon_sha, http_binding, read_jsonl, safe_name,
                                                  suite_record, unique_rows, write_jsonl)


def post(url, messages, max_tokens, seed=1234, model=None, timeout=3600.0):
    body = {"messages": messages, "temperature": 0, "top_p": 1, "seed": seed, "max_tokens": max_tokens,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    if model:
        body["model"] = model
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data, time.perf_counter() - t0


def one(url, arm, item, warmup=False, binding=None, **kw):
    rec = {"id": item["id"] + ("#warmup" if warmup else ""), "arm": arm, "group": item.get("group"), "warmup": warmup,
           "reply": None, "usage": None, "latency_s": None, "strix": None, "error": None,
           "binding": binding, "item_sha256": canon_sha(item)}
    try:
        msgs = item["warmup"] if warmup else item["messages"]
        data, dt = post(url, msgs, item["max_tokens"], **kw)
        rec.update(reply=(data.get("choices") or [{}])[0].get("message", {}).get("content") or "",
                   usage=data.get("usage"), latency_s=round(dt, 4), strix=data.get("strix"))
    except Exception as e:  # recorded, scored as a failure
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def run(url, items, arm, out, concurrency=1, seed=1234, model=None, timeout=3600.0, manifest=None, suite="tasks"):
    binding = http_binding(manifest, arm, suite)
    if suite_record(items) != manifest["protocol"]["suites"][suite]:
        raise ValueError("items/checks differ from frozen suite")
    if seed not in manifest["protocol"]["seeds"]:
        raise ValueError("seed is not in frozen protocol")
    kw = dict(seed=seed, model=model, timeout=timeout, binding=binding)
    results, groups, plain = [], {}, []
    for it in items:
        (groups.setdefault(it["group"], []) if it.get("group") else plain).append(it)
    if plain:
        if concurrency > 1:
            with ThreadPoolExecutor(concurrency) as ex:
                results += list(ex.map(lambda it: one(url, arm, it, **kw), plain))
        else:
            results += [one(url, arm, it, **kw) for it in plain]
    for gid, grp in groups.items():
        for it in grp:
            if it.get("warmup"):
                results.append(one(url, arm, it, warmup=True, **kw))
        n = max(it.get("concurrency", 1) for it in grp)
        if n > 1:
            with ThreadPoolExecutor(n) as ex:
                results += list(ex.map(lambda it: one(url, arm, it, **kw), grp))
        else:
            results += [one(url, arm, it, **kw) for it in grp]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / f"results.{safe_name(arm)}.jsonl", results)
    return results


def prefix_hit(strix):
    """Best-effort: first receipt field whose name mentions prefix and hit/restore/cached; None when absent."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                lk = k.lower()
                if "prefix" in lk and any(w in lk for w in ("hit", "restor", "cached", "reuse")):
                    return v
                r = walk(v)
                if r is not None:
                    return r
        return None
    return walk(strix) if strix else None


def route_measured(item, result, warmups):
    if item["kind"] != "retrieval":
        return True
    meta = item["meta"]
    route = (result.get("strix") or {}).get("qualification_route") or {}
    if not meta.get("measured") or not meta.get("cut_ok"):
        return False
    expected = {"mode": item["mode"], "concurrency": item["concurrency"], "cut": item["cut"],
                "chunk": meta["chunk"], "needle_start": meta["needle_start"], "needle_end": meta["needle_end"]}
    if any(route.get(k) != v for k, v in expected.items()):
        return False
    if item["mode"] == "restored":
        warm = warmups.get(item["group"], [])
        return bool(warm and all(not w.get("error") and w.get("reply") is not None for w in warm)
                    and route.get("prefix_restored") is True)
    return route.get("prefix_restored") is False


def score(items, results_dir, out_dir=None, manifest=None, suite="tasks"):
    by_id = unique_rows(items)
    if suite_record(items) != (manifest or {}).get("protocol", {}).get("suites", {}).get(suite):
        raise ValueError("items/checks differ from frozen suite")
    if any(it["kind"] == "code" for it in items):
        tasks.require_code_sandbox()
    out_dir = Path(out_dir or results_dir)
    written = {}
    for rp in sorted(Path(results_dir).glob("results.*.jsonl")):
        raw = read_jsonl(rp)
        unique_rows(raw)
        main = [r for r in raw if not r.get("warmup")]
        if {r["id"] for r in main} != set(by_id):
            raise ValueError("result coverage differs from frozen suite")
        arms = {r["arm"] for r in raw}
        if len(arms) != 1:
            raise ValueError("mixed arms in results")
        arm = arms.pop()
        binding = http_binding(manifest, arm, suite)
        warmups = {}
        expected_warm = {it["id"] + "#warmup" for it in items if it.get("warmup")}
        if {r["id"] for r in raw if r.get("warmup")} != expected_warm:
            raise ValueError("warm-up coverage differs from frozen suite")
        for r in raw:
            key = r["id"].removesuffix("#warmup") if r.get("warmup") else r["id"]
            if r.get("binding") != binding or r.get("item_sha256") != canon_sha(by_id[key]):
                raise ValueError("stale result binding or changed item/check")
            if not r.get("error") and (r.get("strix") or {}).get("qualification_identity") != binding["server_identity"]:
                raise ValueError("missing or mismatched effective server identity")
            if r.get("warmup"):
                warmups.setdefault(by_id[key]["group"], []).append(r)
        rows = []
        for r in main:
            it = by_id[r["id"]]
            ok = bool(r.get("reply") is not None and not r.get("error") and tasks.score_item(it, r["reply"]))
            row = {"id": it["id"], "kind": it["kind"], "split": it["split"], "arm": r["arm"], "ok": int(ok),
                   "latency_s": r.get("latency_s"), "prefix_hit": prefix_hit(r.get("strix")), "error": r.get("error"),
                   "binding": binding, "item_sha256": canon_sha(it), "server_identity":
                   (r.get("strix") or {}).get("qualification_identity"),
                   "measured": route_measured(it, r, warmups)}
            for k in ITEM_FIELDS:
                if k in it:
                    row[k] = it[k]
            rows.append(row)
        path = out_dir / ("scored." + rp.name[len("results."):])
        write_jsonl(path, rows)
        written[str(path)] = (sum(x["ok"] for x in rows), len(rows))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--items", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--model")
    r.add_argument("--timeout", type=float, default=3600.0)
    s = sub.add_parser("score")
    s.add_argument("--items", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--out")
    for p in (r, s):
        p.add_argument("--manifest", required=True)
        p.add_argument("--suite", choices=("tasks", "retrieval"), default="tasks")
    a = ap.parse_args(argv)
    items = read_jsonl(a.items)
    if a.cmd == "run":
        res = run(a.url, items, a.arm, a.out, a.concurrency, a.seed, a.model, a.timeout, mf.load(a.manifest), a.suite)
        print(len(res), "requests;", sum(1 for x in res if x["error"]), "errors")
    else:
        for p, (ok, n) in score(items, a.results, a.out, mf.load(a.manifest), a.suite).items():
            print(f"{p}: {ok}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
