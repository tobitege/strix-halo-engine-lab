# DS41 prefix RAM ownership

DS41 `PrefixCache` retains RAM blocks and snapshots in the existing
`PrefixStateCache`. The family owning node directory, snapshot clock, byte
ledger, victim selection and ancestor pruning are removed. The facade keeps
layer layout, token-chain verification, boundary eligibility, byte packing and
assembly. Its remaining sequence counter preserves durable file sequence bytes;
it does not order RAM victims.

Each immutable segment contains one block's KV/index rows and a shared reference
to its preceding segment. Each checkpoint retains one leaf and its fixed
ring/carry/next-token/logit metadata. Branches share their common segments;
snapshots never copy the full prefix or encode a full attention state. The
common directory owns checkpoints; the common segment index is weak and carries
retention reference counts. Erasure invalidates that index before releasing
payload ownership. Outstanding immutable read leases retain their byte credits.

The explicit chain policy selects snapshots in scoped LRU order, then releases
unneeded blocks. Admission pins existing ancestry while removing snapshots,
including an ancestor snapshot of a delta insertion. Duplicate insertion and
`use()` touch snapshots; `lookup()`, `covered()` and `load()` do not. Hashes alone
never establish a token match. Unknown-next whole hits, minimum remaining rows,
alignment and missing delta ancestry retain the existing refusal rules.

Legacy block and snapshot charges include their existing bookkeeping allowance.
The shared ledger reserves the complete incremental charge before allocating
payloads, then transfers credits to immutable allocation receipts. Shared blocks
are charged once. Leases and partially prepared captures consume that same service
budget. A failed allocation publishes no partial checkpoint or segment directory;
already selected victims remain evicted, as with the original allocation order.
Zero family budget remains unlimited, without an eager backing allocation. An
optional defaulted owner argument permits independent scoped facades in one
store. Chain admission evicts only its own scope and refuses if other scopes or
outstanding leases consume the remaining service credits. Legacy-mode stores and
attached fixed slabs refuse this policy; ordinary default construction uses the
non-slab shared owner. Existing tiled, named and typed owner defaults are unchanged.

The family disk descriptor inventory, quota policy and async Writer remain.
The [existing durable IO adoption](checkpoint-ds41-prefix-adoption.md) supplies
the same codecs, filenames, locks, restart scan and publication mechanics.
`clear_ram()` leaves disk descriptors and files; `clear()` removes both tiers.
Disk promotion goes through the shared RAM owner. Durable directory/policy
consolidation is unfinished and no second shared disk implementation is added.

The [host fixture](../../tests/ds41_prefix_owner/README.md) checks actual family
and common owner sources, byte charges, ancestry, lifetime, LRU and failure
boundaries. Host evidence does not qualify native arithmetic, physical model
restore, paired execution, unchanged device capacity or performance. This slice
does not complete the durable-directory consolidation, GLM adoption or native
copy-on-write, fork and append support. Numerical identities and production
defaults are unchanged.
