# Image input

Image input lets a chat request carry pictures. Today one model takes them: DeepSeek V4.1 on the two-node pair.
[One shared vision runtime](shared-cores.md#the-components) serves every family; a model only supplies an adapter
([vision-adapters.md](vision-adapters.md) describes the adapter interface).

| Model | Image input |
|---|---|
| DeepSeek V4.1 | yes, on the pair, behind `--vision` |
| Qwen, GLM, MiniMax, Gemma | not yet |

## Use

Start the server with `--vision <vision GGUF>` on rank 0, `--prefill-path device` and an explicitly sized
`--prefill-rows N`. Vision with `--deploy`, `--deploy-check` or `--prefill-rows auto` is refused until the admission
estimates include retained source bytes, device peaks, cached/frame rows and transient copies. Manual sizing must
leave room for these allocations on both ranks. The flag is off by default; without
it no vision code runs and rank 1 holds nothing of it. Send an OpenAI-style chat completion with an `image_url` part
(`data:` URL):

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'
```

## Data path

1. **Media front door** (`ApiMediaBridge`, `include/strix/vision/api_media.hpp`). Decodes PNG and JPEG under
   `DecodeLimits` (`image.hpp`). WebP and GIF are refused with a 400 ("send PNG or JPEG").
2. **Preprocessing** (`preprocess.hpp`, `ResizeRule`). The upstream resize plan: the aspect ratio is kept on a grey
   canvas, sides are multiples of 14, pixels are BF16, patches are merged 3x3 with zero-padded partial cells. The
   span (`embed_span.hpp`, `SpanLayout`) is at most 1024 rows, counting the row ends, the start and the end markers.
3. **Shared encoder runtime** (`Encoder`, `encoder.hpp`; `src/vision/encoder.hip`). One data-driven ViT tower for every
   family. The HIP arithmetic is `vision-f16-v1`: FP16 panels with a power-of-two scale and FP32 accumulation.
   DeepSeek uses split hi+lo activations.
4. **Projector** (the aligner), in the adapter (`VisionAdapter`, `adapter.hpp`; DeepSeek: `deepseek41/vision_adapter.hpp`).
5. **Splice**. The embedding rows enter the token stream as virtual ids. Image rows carry the router's image flag. For
   Engram, image positions take no part in n-grams and the Engram gate is 0 there, as in the upstream model.
6. **Pair frame** (`include/strix/pair/embed_frame.hpp`). Rank 0 encodes; for each new image it sends rank 1 one frame:
   manifest, BF16 rows and a CRC32C.
7. **Embed cache** (`EmbedCache`, `embed_cache.hpp`). Key = adapter identity + arithmetic + image digest. LRU, default
   256 MiB (`STRIX_VISION_CACHE_MIB`; `0` disables storing). It lives for the process only and is never written to
   disk. Image prompts never enter the prefix cache or sessions.

## Measured

On the two-node pair, DeepSeek V4.1 Q2, one 37x23 PNG fixture upscaled to 1550 patches (200 image rows + 21 text
tokens):

| Case | Time to first token | Detail |
|---|---|---|
| First request (image not cached) | 1.46-1.87 s | image encode 0.93-1.07 s; frame to rank 1 2.0 MB in 4-5 ms; LM prefill about 0.46 s |
| Same image again (cache hit) | 0.53-0.61 s | encode skipped; 80-byte frame |

Decode: 39-41 tok/s.

## Switches

| Switch | Default | Meaning |
|---|---|---|
| `--vision FILE` | off | Server flag, rank 0 only. Needs `--prefill-path device`. |
| `STRIX_VISION_RESIDENT` | 1 | `1`: the tower is uploaded at start; `0`: on the first image. |
| `STRIX_VISION_CACHE_MIB` | 256 | Embed cache size; `0` disables storing. |

See [toggles](../toggles.md).

## Limits

- PNG and JPEG only.
- Images over the row budget are refused.
- Image requests with `session_id` are refused.
- Video is refused.
- One image adapter so far (DeepSeek V4.1); other models do not take images yet.
- Needs the device prefill path.
- Encoding is serialized with media preprocessing; peer/split scheduling and overlap with text-prefix prefill remain future work.
- Image positions have zero Engram gates. Synthetic gate tests do not qualify complete served-image fidelity;
  qualification evidence must name the tested revision and cover that arithmetic in the complete image workload.
