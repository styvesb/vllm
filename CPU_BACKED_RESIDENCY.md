# Experimental CPU-backed Qwen MoE milestone

Runtime implementation: `33eda19` (loading) and `b57c2b4` (live demand-only
execution), based on upstream `5df63a7203cdb9328d824a27b682e77c7b9928be`.
The companion project at `../../RESIDENCY.md` contains setup and reproduction
instructions; project execution implementation is preserved at `d794c86`.

Audited 2026-11-03: companion artifacts
`experiments/live_full_model_v4/results.json` and `generation.json` agree on
output `" Paris."`, tokens `[12095,13]`, all 48 layers, 96 CPU expert parameters,
75,497,472 cache bytes, 525 hits, 2,163 misses, 2,155 evictions, and zero full
expert GPU parameter bytes outside the cache. Counters include startup work.
The JSON evidence is ignored in the companion repository. The reported temporary
log is absent at audit, so the supplied process-exit report is not independently
rechecked here. No GPU workload was rerun for this preservation step.

Six earlier real complete-block comparisons were bit-identical to normal
resident vLLM. Full-model numerical equivalence and performance gains remain
unestablished. Supported execution is single-GPU/request, canonical BF16 Qwen3,
eager, demand-only. Speculation is still rejected. See the companion
`PREDICTOR_INTEGRATION_PLAN.md` for a design requiring approval before coding.

Use the isolated `.venv-residency` interpreter and existing project cache;
original installed vLLM and `.venv` are preserved. This documentation commit
changes no runtime behavior.

Milestone 1 adds optional proposal identity metadata under
`VLLM_RESIDENCY_PROPOSAL_IDENTITY=1`. It transports an immutable-by-convention
primitive envelope through DraftTokenIds, scheduler Request state and
SchedulerOutput, then checks the actual prepared tokens/positions in the V2
runner. Default tracing is off. Token acceptance, cache decisions, pass ownership,
and the speculative-runtime exclusion are unchanged. CPU fake-batch coverage is
in the companion project's `tests/test_proposal_identity.py`; GPU EAGLE integration
is not validated. See companion RESIDENCY.md for the inspected path and limitations.
