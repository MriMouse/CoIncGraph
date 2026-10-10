# Dynamic CPU–GPU CC

hybrid_cc matches old_hybrid_cc: every vertex starts with its own ID and
propagates its minimum label along the input directed edges. The result is
label[v] = min { u | u can reach v }, including v itself. Input and update
weights are ignored. Edges are used as supplied; no reverse edges are added.
This is directed minimum-label propagation, not strongly connected components.

The application retains the shared chunk store, reverse index, GPU deletion
repair, block/ordered insertion, hotness/cache maintenance
and batch timers. CC specializes the existing operators: vertex seeds, zero
traversal cost and equal-label deletion dependencies. Deletion repair resets
affected vertices to their own IDs; rooted witnesses avoid redundant work.
After each complete deletion phase, bounded forward traversal from actual label
roots creates shared proof anchors. Reverse queries connect to these anchors,
avoiding ID-ordered blind searches inside large equal-label cycles. Certificates
are local to that phase and never survive an insertion or another deletion batch.
`CG_CC_FORWARD_WITNESS=0` disables the forward anchors for comparison;
`CG_CC_ROOTED_WITNESS=0` disables the complete witness fast path. Exhaustion
always falls back to conservative invalidation and exact repair.

Build: cmake --build build --target hybrid_cc -j2

Flags: --cc_max_batches, --cc_print_checksum.
--source_node is accepted for compatibility and ignored. Static runs may omit
both update paths; weighted update records require --weight=true, with their
numeric weights ignored. --output writes two columns: vertex label.
--check=true compares labels to an independent directed flood-fill oracle after
each deletion/insertion phase and at completion. This diagnostic work is outside
the batch timer; performance runs should use --check=false.
