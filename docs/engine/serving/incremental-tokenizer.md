# Incremental tokenizer and early prompt admission

A long prompt does not have to be fully tokenized before its first prefill chunk starts. The tokenizer can
yield ids in order, each one final as soon as it is yielded, and the session runtime can start on the ids it
holds while the rest is still being encoded on the [host lane](../scheduling/host-lane.md).

## Incremental encoding

`BpeTokenizer::encode_some(text, parse_special, cursor, slice_bytes, out)` encodes about `slice_bytes` more
of `text` per call and appends the ids. The ids appended over all calls are exactly
`encode(text, parse_special)`; nothing appended is ever revised.

A slice ends only where the two sides are encoded independently by construction:

- at a special token, which the one-shot encoder also treats as a hard separator;
- between an ASCII letter and a following space. In every implemented pre-tokenizer the letter ends a
  piece there whatever follows, no earlier piece reads past the space, and the scanners never look behind
  a piece start. BPE merges are applied per piece, so no merge crosses the boundary.

A text with no such place (for example one long run of CJK text) is encoded in one final step: correct,
only not early.

`ApiTokenizer::encode_incremental()` exposes the same thing behind the serving tokenizer interface. Its
default implementation yields the one-shot encoding in a single step, so every tokenizer supports it.

## Prompt stream

`PromptStream::start(tokenizer, text)` runs the incremental encoding as critical host-lane jobs and
publishes the final ids. `wait_for(n)` blocks until `n` ids are final; `take(prompt, wait)` appends the ids
that became final to a prompt that holds the earlier ones.

## Early admission in the session runtime

A `GenerationJob` may carry `prompt_tail`: `prompt` then holds the ids that are final so far and the tail
supplies the rest. The runtime

- appends newly final ids before each scheduling round;
- executes a prefill chunk for such a lane only when the chunk lies strictly inside the ids it holds, so a
  plan never depends on where the held ids happen to end; otherwise it waits for more ids and plans again;
- never treats the prompt as complete before the tail is, so decode, prompt checkpoints and cache capture
  see the whole prompt exactly as before.

Chunk boundaries, consumed ids and output are the same as for the complete prompt.

When a prefix or turn cache is active, or the turn is owner-controlled, the lookup decides on the whole
prompt: the runtime then completes the prompt at admission, and the encoding overlaps only the time the
request spent queued. Jobs without `prompt_tail` are untouched.

## Tests

- `strix-incremental-tokenizer-tests`: for the three pre-tokenizers, with and without special parsing, on
  a generated corpus (words, contractions, digit runs, whitespace and newline runs, punctuation, CJK,
  combining marks, malformed UTF-8, special tokens including ones with a letter and a space inside): the
  yielded ids are a prefix of the one-shot ids after every step for slice sizes from 1 byte up, every
  slice end splits the text into two independently encoded parts, and the final ids are equal. Also the
  default one-step form and the prompt stream with the lane on and off.
  `--bench TOKENIZER [WORDS]` prints the one-shot time and the time until the first chunks are final.
- `session_runtime_tests`: an open prompt produces the same prefill chunks and output as the complete
  prompt for several head sizes and growth steps; with a cache the tail is settled at admission; a failing
  tail fails only its request.
