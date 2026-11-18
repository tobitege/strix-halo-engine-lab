# Ollama-compatible API

Every service built on the shared API layer (`src/serve/api.cpp`, HIP node, USB4 pair front and CUDA alike) also
answers the Ollama HTTP contract. The adapter (`src/serve/ollama.cpp`) has no inference path of its own: it rewrites
the request into the OpenAI-shaped body the chat / completions handlers take, runs it through them, and re-encodes
their output. Prompt rendering, token IDs, admission, cancellation, metrics and the request ring are therefore the
ones of the equivalent OpenAI request. On by default; `STRIX_SERVE_OLLAMA=0` (or `ApiConfig::ollama_api = false`)
turns the routes off. They take the same API key as the other routes (`Authorization: Bearer` or `x-api-key`).

| Route | Runs as | Notes |
|---|---|---|
| `POST /api/chat` | `/v1/chat/completions` | `messages` (`role`, `content`, `thinking`, `images`, `tool_calls`, `tool_name`), `tools`, `format`, `options`, `think`, `stream` (default `true`) |
| `POST /api/generate` | chat with one user message (+ `system`); `raw: true` runs `/v1/completions` | `suffix`, `template` and `context` are refused (they would change the prompt) |
| `GET /api/tags`, `GET /api/ps` | metadata | the one resident model; `ps` reports `context_length` |
| `POST /api/show` | metadata | `capabilities` lists only what the engine executes: `completion`, `tools`, `thinking` (template default), `vision` (image adapter loaded) |
| `GET /api/version` | metadata | the API level implemented (`ApiConfig::ollama_api_version`) |
| `/api/embed`, `/api/embeddings` | 501 | no embedding route is executed by these engines |
| `/api/pull`, `push`, `create`, `copy`, `delete`, `blobs` | 501 | one resident model; no model management |

A request with no `messages` / `prompt` is a load probe: it answers `done_reason: "load"` and runs nothing.
`model` must name the served model (`/api/tags`), optionally with `:latest`; any other name is 404.

## Options

| Ollama option | Becomes | |
|---|---|---|
| `temperature`, `top_p`, `top_k`, `min_p`, `seed` | same OpenAI field | validated by the shared sampling resolver |
| `repeat_penalty`, `presence_penalty`, `frequency_penalty` | `repetition_penalty`, `presence_penalty`, `frequency_penalty` | refused where the engine has no logit transforms (as on the OpenAI route) |
| `num_predict` | `max_tokens` | `-1` / `-2`: the server default bound |
| `stop` | `stop` | string or array |
| `num_ctx` | per-request context bound | below the served context it bounds prompt + output (prompt larger: 400, never truncated; output stops with `length`); at or above it: no effect |
| `typical_p`, `tfs_z` != 1, `mirostat` != 0 | 400 `unsupported_option` | they change the result and are not executed |
| `repeat_last_n`, `penalize_newline` | no effect while the penalties are neutral, else 400 | |
| `num_batch`, `num_gpu`, `main_gpu`, `num_thread`, `use_mmap`, `use_mlock`, `low_vram`, `numa`, `num_keep`, `keep_alive` | no effect on the committed tokens | listed in the `X-Strix-Ollama-No-Effect` response header |
| anything else | 400 `unknown_option` | nothing is accepted silently |

`format: "json"` becomes `response_format: json_object`; a schema object becomes `json_schema` (the constrained
decoding of the OpenAI route). `think: true/false` sets `enable_thinking`; `"low"|"medium"|"high"` sets
`reasoning_effort`. Images (raw base64 PNG / JPEG / GIF / WebP) become `image_url` parts and are refused when no
image adapter is loaded. Defaults for options the request leaves out are the server's (the same as an OpenAI
request without them), not another server's model file.

## Responses

Non-streamed: one JSON object with `message` (`content`, `thinking`, `tool_calls` with object `arguments`) or
`response`, and `done`, `done_reason`, `prompt_eval_count`, `eval_count`, `total_duration`, `prompt_eval_duration`,
`eval_duration` (ns; from the engine's prefill / decode times when it reports them), `load_duration: 0` (weights
are resident). Streamed: `application/x-ndjson`, one line per delta, tool calls as one complete line, then the
`done` line. Errors are `{"error": "..."}` with the HTTP status; an error during a stream is one error line.
SSE keepalive comments have no NDJSON form and are not sent; a client that disconnects still cancels the generation.

`strix-serve-api-tests` reports the host cost of translating one streamed token frame. This microbenchmark
does not measure full HTTP/decode overhead or establish a percentage of end-to-end generation time.
