# MiniMax shared HTTP frontend

`MinimaxApi` composes `serve::ApiService` over the existing `GenerationQueue`.
`handle` only forwards the request and response. The shared service owns JSON
normalization, HTTP envelopes, SSE delivery, UTF-8 handling, disconnect and
cancellation cleanup. The existing MiniMax planner and `SessionRuntime` retain
one external ticket and the physical slot; no additional listener, queue,
request owner or cache is installed.

The family adapter supplies the actual MiniMax tokenizer and chat renderer,
final reasoning parsing, capability refusals, ordered timing data and observation
data. Encoding, special-token policy, decoding, lookup, vocabulary, token type
and EOS metadata delegate to the family tokenizer. Existing callback-tokenizer
callers remain supported; unavailable optional metadata refuses only operations
that need it, before generation admission. Nonstream generation uses the encode
and decode callbacks; streaming additionally requires the stream-bytes callback.

| Route | Contract |
|---|---|
| `/v1/chat/completions` | Existing compact nonstream and SSE envelopes, final reasoning split, content-only live streaming, optional generated IDs and usage chunk. |
| `/v1/completions` | Raw string prompt, existing text-completion envelopes and live SSE, optional IDs and usage chunk. |
| `/v1/messages` | Shared message normalization and Anthropic envelopes; same family rendering and execution owner. Explicit thinking mode maps to the family template. |
| `/v1/responses` | Shared stateless normalization and Responses envelopes; same family rendering and owner. Reasoning effort remains a family template value. |
| `/tokenize`, `/detokenize` | Actual family tokenizer; legacy `text` input and token-array envelope, chat rendering for message input, vocabulary-checked detokenization. |
| `/health`, `/v1/health`, `/v1/models`, `/v1/capabilities` | Shared routes with family identity and truthful policy data; historical minimal health/model presentation. |
| `/api/status`, `/metrics` | Existing family observer, recent-request ring and `mm_*` / `strix_*` metrics; shared route and arming lifecycle. |
| Deployment and `/admin/drain` | Existing shared deployment callbacks and queue admission drain. |
| `/admin/sched` | Existing optional scheduler callback; shared framing, original enablement and domain checks, and the existing 400 error contract for callback exceptions. |

Equivalent normalized chat, messages and Responses inputs render and encode
once into the same family prompt IDs. An absent thinking control stays absent;
explicit template controls survive normalization. Automatic/none tool choice
and family tool history remain template operations. Required/named choices,
strict tool enforcement, constraints, logprobs, logit transforms, thinking
budgets, request draft overrides and nonempty stop strings are refused by name
before queue work. HTTP `state_cache` overrides are also refused; retention
policy stays with the existing owner. Zero/false logprobs retain their legacy
neutral meaning. Route presence does not imply those policies execute.

Decoding remains greedy. Positive temperature and nondefault greedy filters
retain their declared ignored-policy notes. Maximum token counts are parsed
from exact JSON integer spellings, with `max_completion_tokens` preferred and
clamping to the full configured context rather than wrapping. Generated ID
receipts and usage include EOS. Native session identifiers retain their existing
validation, slot owner and fixed execution policy; session administration and
cache invalidation remain in the existing service/runtime.

After every accepted ticket reaches terminal, the shared lifecycle drains
`take_extra` exactly once, including queued cancellation, failed stream start,
failed writes, disconnect and execution errors. The terminal hook returns domain
data and never receives a writer. A separate completion hook records delivery
once through the existing observer, replacing the shared recent-request owner.
Validation and queue rejection take no scheduler receipt. A rejected queue
request may retain its existing observed refusal record. Valid inbound request
IDs are echoed; generation uses its completion ID as the fallback.

Host tests cover production frontend/queue/runtime/planner adapters using a real
family tokenizer and mutable host target rows, independently frozen HTTP replay,
terminal cleanup and existing service/session suites. Native compiler/link,
both-rank token/logit/state fidelity, unchanged capacity and complete-workload
performance remain separate qualification gates. See the
[session contract](session-layer.md) and [host project](../../tests/minimax_shared_api/README.md).
The next frontend adopters are Qwen, DS41 and GLM. Generic resource, recurrent,
parallel and page implementations without a production adopter remain held.
