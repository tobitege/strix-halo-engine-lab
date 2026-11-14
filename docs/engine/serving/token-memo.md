# Prompt token memo

A chat request resends its whole history. Without a memo the server renders the template and encodes all of
that text again on every turn, before the prefix cache is consulted. The memo keeps the text and the ids of
recent prompts and encodes only what is new.

## How it stays exact

The tokenizers scan a text from left to right and split it at special (control / added) tokens; the text
between two of them is encoded on its own. The end of a special token is therefore a *restart point*: nothing
is carried across it. `encode_marked()` returns the ids together with these points (`EncodeMark`: byte offset,
ids before it).

For a new text the memo finds the kept text with the longest shared byte prefix and takes the last restart
point whose offset plus `mark_lookahead()` (the longest special token) lies inside the shared bytes. The window
matters: with the tokens `<a>` and `<a>b`, a text that continues with `b` right after `<a>` matches the longer
one, so a restart point is only used when every match that starts before it is decided by shared bytes. The
ids before the point are copied, the rest of the text is encoded, and the result is the full encoding.

The memo compares bytes, never messages: an edit in the middle, reordered tools, a changed system prompt or a
different template simply shorten the shared prefix. A tokenizer that reports no restart points gets
whole-text hits only.

## Scope and bounds

- Shared: `strix::serve::TokenMemo` (`include/strix/serve/token_memo.hpp`). The API service uses it for the
  rendered chat prompt and for the stable system/tools prefix; the MiniMax API uses it through its tokenizer
  callback. `BpeTokenizer` (Qwen, DeepSeek, GLM) and `MinimaxTokenizer` report restart points.
- Key: retained tokenizer lifetime token, special-token parsing and caller salt, compared field by field.
  Construction, copying and assignment start a fresh lifetime; configuration must otherwise stay immutable.
  The callback overload requires the caller to supply a unique immutable vocabulary/configuration identity.
- Memory: 64 MiB and 32 entries by default, least recently used entry evicted first. A prompt that continues a
  kept one replaces it, so a conversation holds one entry; conversations that share a prefix keep their own.
  The resident-entry charge includes string/vector capacities, the string terminator, entry storage and a
  bookkeeping allowance for list nodes and shared ownership. It is not a process-memory cap: allocator-specific
  metadata, in-progress encodes, returned ID copies and pinned evicted entries consume memory outside this budget.
- Thread-safe; encoding runs outside the lock.

| Variable | Default | Meaning |
|---|---|---|
| `STRIX_SERVE_TOKEN_MEMO` | `1` | `0` encodes every prompt in full |
| `STRIX_SERVE_TOKEN_MEMO_MB` | `64` | byte bound of the memo |
| `STRIX_SERVE_TOKEN_MEMO_ENTRIES` | `32` | entry bound of the memo |

## Proof

`strix-token-memo-tests` compares the memo with a full encode on random chats through the three chat
templates (turns appended, edits in the middle, tools reordered or changed, system prompt changed, sessions
forked from a shared prefix, eviction under a small bound), on raw texts cut at any byte (inside UTF-8
sequences and inside special tokens), without special parsing, across tokenizers and from eight threads.
`--bench TOKENIZER FAMILY` prints host medians for the same before/after history, alternating full/memo
measurement order and resetting the memo to the same predecessor before each pair. These host samples do
not qualify concurrent contention or served time to first token.
`strix-minimax-tokenizer-tests` runs memo checks on a small byte-vocabulary fixture when the optional model
tokenizer is absent; model-specific golden-ID checks still require that asset.
