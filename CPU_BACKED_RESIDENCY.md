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

Milestone 2: supported speculation is synchronous, fixed two-token BF16
`LlamaForCausalLMEagle3` with standard rejection sampling, no adaptive/dynamic
proposal sizing and the original target restrictions. The draft has dense MLPs
and uses normal GPU storage; only target RoutedExperts use the shared cache.
Optional identity tracing records bounded alignment events and basic proposal /
verification / acceptance counts. No Ridge, cache-policy or pass-ownership changes.

Approved companion smoke `experiments/eagle_demand_smoke_v3` passed on L40S:
7 proposals, 14 proposed tokens, 6 verification passes, 12 verified draft tokens,
5 accepted draft tokens; all 5 supported identity envelopes matched. All 48 target
layers ran, 96 full expert parameters remained CPU-backed, and the shared cache
used 75,497,472 bytes. Cache hits/misses/evictions: 2254/6962/6954 (includes startup).
The draft had no CPU-backed method or cache. Report errors were empty. Both
processes exited; GPU memory was released. See companion RESIDENCY.md for limits.

Milestone 3 adds opt-in `VLLM_RESIDENCY_OBSERVE=1` feature capture and NumPy Ridge
observation via the companion project's `RidgeObserver`. It requires the existing
identity tracer and supported two-token eager EAGLE baseline. Draft model sample
hidden states are cloned at steps 0 and 1, then copied together and associated
with the existing envelope. Matching verification consumes/discards scores;
no scores enter the cache/scheduler. Default mode remains disabled.

Approved observation smoke passed: 6 supported captures/predictions, 5 matching
verifications, 1 unused prediction discarded, one validated artifact load and no
pending result or report errors. Output and cache counters matched the saved
Milestone 2 smoke. Median copy/preprocessing/projection wall times over 6 samples:
0.072/0.067/2.552 ms. Full details, timing limits, CPU coverage and the exact
scikit-learn preprocessing compatibility correction are in companion RESIDENCY.md.
No guided admissions, residency changes or pass-ownership changes were added.

Milestone 4 adds default-off `VLLM_RESIDENCY_GUIDED=1`. The V2 runner consumes a
validated prediction before target execution and owns guided pass begin/end;
MoE layers advance that existing pass only. Demand/observation retain their old
layer-owned lifecycle. Missing/profile/prefill predictions use demand-only passes;
exceptions/request removal abort active guided state. Existing four-layer K=8
policy, locks, budget, mandatory routing and synchronous transfers are retained.

Approved one-load/cold-cache three-mode correctness comparison passed with
identical output token IDs and fixed 75,497,472-byte slot pointers. Demand and
observation: 6578 mandatory loads, zero speculative loads. Guided: 1920 predicted
admissions plus 5667 mandatory loads, 7587 total. All load/byte accounting balanced;
48 target layers ran with CPU-owned full expert tensors and no full expert GPU
parameters. Five guided passes and two demand fallbacks completed without resets.
Total transfers increased in this request; no performance benefit is claimed.
See companion RESIDENCY.md and experiments/guided_comparison_v1 for details.
