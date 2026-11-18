#!/usr/bin/env python3
# Vision battery client: one greedy chat request per image (data URL) to a served engine; writes a JSON list that
# score.py grades against facts computed from the pixels.
# usage: battery.py PORT OUT.json TAG IMG...   env BATTERY_MODEL (model field, default deepseek-v4.1), BATTERY_MAX_TOKENS (80)
import base64, json, os, sys, time, http.client

PROMPT = ("Describe this image in one or two sentences: which colours and shapes or patterns do you see, "
          "and where are they in the image?")
def run(port, out, tag, imgs):
    manifest = [{"name": os.path.splitext(os.path.basename(img))[0], "image": img} for img in imgs]
    with open(out + ".manifest.json", "w") as stream:
        json.dump({"requested": manifest, "terms": tag}, stream, indent=1)
    res = [{"name": item["name"], "terms": tag, "status": "pending", "error": "not completed"} for item in manifest]

    def persist():
        with open(out + ".tmp", "w") as stream:
            json.dump(res, stream, indent=1)
        os.replace(out + ".tmp", out)

    persist()  # interrupted clients retain the entire requested denominator
    for i, img in enumerate(imgs):
        name = manifest[i]["name"]
        c = None
        t0 = time.time()
        try:
            with open(img, "rb") as stream:
                raw = stream.read()
            mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
            body = {"model": os.environ.get("BATTERY_MODEL", "deepseek-v4.1"), "stream": True,
                    "max_tokens": int(os.environ.get("BATTERY_MAX_TOKENS", "80")), "temperature": 0,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode())}}]}]}
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=float(os.environ.get("BATTERY_TIMEOUT_S", "40")))
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            response = c.getresponse()
            if response.status != 200:
                raise RuntimeError("HTTP %d: %s" % (response.status, response.read()[:300].decode("utf-8", "replace")))
            text, ttft, strix, done = "", None, {}, False
            for line in response:
                line = line.decode("utf-8", "replace").strip()
                if line == "data: [DONE]":
                    done = True
                    break
                if not line.startswith("data:"):
                    continue
                j = json.loads(line[5:])
                if j.get("error"):
                    raise RuntimeError(str(j["error"]))
                for ch in j.get("choices", []):
                    delta = ch.get("delta", {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.time() - t0
                        text += delta
                strix.update({k: v for k, v in j.items() if k.startswith("strix")})
            if not done or not text:
                raise RuntimeError("incomplete/empty stream")
            res[i] = {"name": name, "terms": tag, "status": 200, "ttft_s": ttft,
                      "wall_s": time.time() - t0, "text": text, "strix": json.dumps(strix)[:600]}
        except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
            res[i] = {"name": name, "terms": tag, "status": "error", "error": str(exc)}
        finally:
            if c is not None:
                c.close()
            persist()
        print(tag, name, res[i]["status"], flush=True)
    failed = sum(r["status"] != 200 for r in res)
    with open(os.path.join(os.path.dirname(out), "client-result.json"), "w") as stream:
        json.dump({"completed": len(res) - failed, "failed": failed}, stream)
    return int(failed > 0 or not res)


if __name__ == "__main__":
    sys.exit(run(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]))
