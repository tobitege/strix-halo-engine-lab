# Prefill compression: region adapters

Prefill compression is a LOSSY, per-request opt-in feature that is OFF by default. This page covers the layer
that tells the selection code which bytes of a rendered prompt it may shorten. Nothing calls it yet: every
request is unchanged.

## What an adapter does

`include/strix/serve/pflash_regions.hpp` has one function per chat family:

```cpp
RegionMap qwen_regions(messages, tools, rendered, options);
RegionMap glm_regions(messages, tools, rendered, options);
RegionMap deepseek_regions(messages, tools, rendered, options);
RegionMap minimax_regions(messages, tools, rendered, options);
```

`messages` and `tools` are the request values given to the family renderer and `rendered` is the renderer's
output. The adapter does not render. It walks the messages in order and looks for each message text in the
rendered prompt, accepting a position only when the bytes around it are the markers that family writes around a
turn of that role. The result is a list of `Region`s (byte ranges of the rendered prompt, ascending and disjoint)
for `segment()` of `prompt_compress.hpp`, one `RegionKind` per region, the index of the query turn, and two
fail-safe signals:

- `unlocated`: messages whose text was not found (media parts, content the template rewrites, markers that do not
  match). They get no region. Bytes outside every region are always kept, so a miss makes the prompt longer and
  never drops anything.
- `tools`: the request has tool definitions, tool calls or tool results. No region is droppable; the caller
  serves the request as rendered (`bypass(raw, Reason::tools)`).

## What is kept and what may be shortened

| Part of the prompt | Region | Droppable |
|---|---|---|
| Role tags, turn envelopes, reasoning blocks, tool-call frames, generation prompt | none | never |
| System / developer / root text | `system` | never |
| Tool definitions block | `tool_schema` (Qwen, GLM, MiniMax); none for DeepSeek | never |
| Tool results | `tool_result` | never |
| Turns after the query turn | `tail` | never |
| User and assistant text before the query turn | `history_user`, `history_assistant` | yes (`compress_history`) |
| Text of the query turn | `query` | yes (`compress_query_turn`) |

The query turn is the last user message. In the Qwen family a user message that is one `<tool_response>` block
is tool traffic and not the query, the rule the template itself uses.

The query turn is droppable by default because a one-shot long prompt carries the document and the question in
the same message. The selection layer then keeps the units that hold the request tokens marked by the scorer
(`Policy::query_pin`) and every message of at most `Policy::skeleton_tokens`. Set `compress_query_turn = false`
to keep the whole last user turn, and `compress_history = false` to keep earlier turns.

## Family notes

| Family | Text as printed | Markers checked around a user text | Notes |
|---|---|---|---|
| Qwen | stripped | `<\|im_start\|>user\n` ... `<\|im_end\|>` | leading system messages are merged after the tools block |
| GLM | as given (assistant answer stripped) | `<\|user\|>` ... next `<\|` tag | a late system message is its own turn |
| DeepSeek | as given | user tag or blank line ... next tag or blank line | text parts and merged user messages are blocks joined by a blank line |
| MiniMax | as given | `]~b]user\n` ... `[e~[` | an assistant message with reasoning inside its content is rewritten by the template and is not located |

## Limits

- The scoring rules behind the published description of this technique (a recall gate for follow-up turns, a
  spread fallback, smoothed scores) are parameters of our own selection layer, not behaviour taken from a
  reference implementation. The adapters only provide regions; they add no rule of their own on top of
  `prompt_compress.hpp`.
- A text that also appears verbatim, with the same markers around it, inside an earlier reasoning block or
  tool-call argument of the same conversation could be located there. Regions are checked to be ordered and
  disjoint, and such a match only moves a region onto identical bytes.
- Nothing is measured here: no scorer runs and no engine is involved.

## Tests

`strix-pflash-regions-{qwen,glm,deepseek,minimax}-tests` render each case with the family's own renderer and
check: exact region bytes per message, flags per option, text that imitates the template, text parts, tool
traffic served raw, counted misses, and a run through `segment` / `select` / `assemble` with a stand-in scorer
where every mandatory text and every turn marker survives.
