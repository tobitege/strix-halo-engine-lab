# Conversation keep (kept lane states for follow-up turns)

A follow-up chat turn repeats the whole conversation in its prompt. Without a kept state the server prefills
all of it again; with one it restores the longest kept prefix and prefills only the new tokens.

## Shared policy

`strix::serve::ConversationKeep` (`include/strix/serve/conversation_keep.hpp`) owns what every family needs and
nothing family specific: longest strict-prefix lookup with a token compare, a byte ledger with a budget, LRU order,
leases (an entry being restored is never a victim), a two-phase store with an ordered victim list that a second
rank can replay, reclaim and shrink plans for memory pressure, and statistics. Families keep their own payload
(KV rows, recurrent state, index tables) under an opaque id. This index owns immutable whole-lane
snapshot retention; `session::PrefixCache` owns certified block/resident-slot lookup through `KvStore`.
They do not share payload ownership or admit each other's entries. Consolidating their eviction policies
needs a separate ownership design before additional adapters are added.

The budget is a capacity term: `keep_budget_grant` / `with_keep_budget` in `include/strix/serve/slot_capacity.hpp`
grant what is left after headroom and the live slots, and the slot report charges the grant before it counts
slots (`keep=` / `keep_bytes`). A smaller grant shrinks the cache, never the slots.

## GLM

The prompt snapshot cache (`--prefix-cache-gib`, default 4 GiB per rank) is an adapter over the shared policy.
In a deployment plan it is the owner `prefix_keep`: elastic by default, mandatory when the operator names a size;
the pair uses the smaller rank grant. Retained entries plus a pending snapshot must fit that grant:
the adapter checks a conservative full-layout size bound before copying, and refuses a store that
would exceed it, without evicting existing entries. Preparation allocates all nodes before voting;
commit transfers their ownership without allocation, and abort destroys the pending payload. When the lead observes host memory pressure at admission, kept entries are evicted before a
request or its snapshot is refused.

A follow-up turn hits only if its prompt extends the kept key. With MTP the key is the prompt plus the first
generated token, and the chat template opens the reasoning block in the generation prompt, so:

- a client that returns the reasoning of earlier turns (`reasoning_content`, with `preserve_thinking: true`) hits;
- a client that drops it renders `<think></think>` in the history and misses under MTP (it hits with `--mtp-spec 0`).

`STRIX_GLM_SERVE_REPLY_KEEP=1` (opt-in, plain decode only) also snapshots the end of a reply. Its follow-up is
not always token-identical to a cold prefill, so it stays off.

Measured on a served pair, three-turn conversation, 48 generated tokens per turn, follow-up turns:

| Case | Follow-up TTFT off -> on | End-to-end tok/s off -> on | Prefill tokens computed |
|---|---|---|---|
| 2K context, one conversation | 5.0-5.6 s -> 0.80-0.82 s | 7.0-7.6 -> 21.9-24.2 | 2,062-2,125 -> 63-68 |
| 8K context, one conversation | 15.1-15.2 s -> 0.77-0.83 s | 2.94 -> 20.5-24.9 | 7,881-7,944 -> 63-68 |
| 2K context, four conversations at once | 10.6-15.4 s -> 1.1-1.2 s | 1.2-2.0 -> 7.6-10.0 per conversation | as above |

Reply text is identical with the cache on and off in every turn of these runs (one and four conversations).

## MiniMax

`STRIX_MM_SERVE_PREFIX=1` reuses group-aligned prefill rows of a resident slot (8192 tokens by default), which
never applies to a 2K or 8K conversation. `STRIX_MM_SERVE_CONVERSATION=1` on top of it (both ranks) keeps the
rows a finished reply left in its slot and restarts the follow-up prefill at that frontier. No extra memory is
kept: the state is the slot itself.

| Case | Follow-up TTFT off -> on | End-to-end tok/s off -> on | Prefill tokens computed |
|---|---|---|---|
| 2K context, one conversation | 4.8-5.0 s -> 0.62-0.64 s | 6.8-7.0 -> 18.6-19.4 | 2,202-2,271 -> 69-75 |
| 8K context, one conversation | 12.3-12.5 s -> 0.60-0.61 s | 3.3-3.4 -> 18.9-19.3 | 7,951-8,020 -> 69-75 |
| 2K context, four conversations at once | 12.2-12.6 s -> 1.5 s | 2.1 -> 8.3-8.6 per conversation | as above |

Conversation mode is **not token-identical** to a cold prefill: follow-up replies differed in every run, and a
first turn differed when it restarted inside rows another conversation left in the slot. It is off by default.
