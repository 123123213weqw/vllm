# DFlash2 tree verification: GDN correctness oracle

This fork-only prototype is a test fixture for
[vLLM #70057](https://github.com/vllm-project/vllm/issues/70057).
It checks one Qwen-style GDN head in CPU float64 using synthetic projections.
It is not a drafter, verifier backend, or throughput benchmark.

## Boundary for collaboration

The tree builder owns candidate selection and emits topology. To exercise GDN
with a generated tree, adapt its parent indices to topological row order:
row 0 is the anchor (parent `-1`), and every other row follows its parent.
Use only actual rows, excluding padding. Gather the projected inputs and
transformed gates in that same order. The committed prefix state must exclude
the anchor. Verification returns post-row recurrent state and convolution
history. Commit consumes the accepted **row IDs**, including the anchor, after
stop/length trimming; the next token picked by the target is not committed
until it has a forward pass.

The tests compare tree snapshots with independent single-path scoring, perturb a
sibling to detect state leakage, and check the next token's computation after
committing a full or truncated path. Full per-node snapshots keep this reference
simple; they say nothing about the production state layout. Integrations using
TreeWY or another state representation should satisfy the same comparisons.

The open [#64844](https://github.com/vllm-project/vllm/pull/64844) proposes
generic MRV2 tree execution hooks. This fixture neither depends on that PR nor
introduces competing hooks. It does not verify attention masks, KV movement,
full-model greedy equality, CUDA Graph replay, or serving performance.

## Run locally

```sh
uv venv --python 3.14
uv pip install --python .venv/bin/python torch==2.15.0 pytest==9.2.0 numpy==2.5.5
.venv/bin/python -m pytest --noconftest -q benchmarks/dflash2_tree/test_gdn_oracle.py
```
