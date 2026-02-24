# Iterative Debug Supervisor Contract v1.1 (Smoke-Gated)

Date: 2026-02-24  
Owner: System-level automation agent  
Status: Partially implemented (v1 smoke-gate shim + local validation complete)

## 1) Goal
Make long experiment execution safe by default:
- catch trivial launch/runtime bugs early with a tiny smoke run
- prevent infinite relaunch loops
- keep recovery bounded and evidence-based
- produce deterministic terminal status with cause and next action

## 2) Why This Is Needed
`job_start + watch.thenTask` gives callback hooks but does not impose a standard safety contract. Failures now split into:
- early trivial bugs that waste full-run time (import/arg/path/schema mistakes)
- repeated retries without strong stop criteria

v1.1 addresses this by enforcing **Stage 0 smoke gate** before any full budget run.

## 3) In Scope / Out of Scope
In scope:
- default long-run launch contract
- smoke gate before full run
- bounded retries and bounded debug loop
- required artifact checks and suspicious-success checks
- mandatory handoff/memory updates per cycle

Out of scope:
- unlimited autonomous code edits
- broad refactors during recovery
- automatic merge to main

## 4) Target Flow (v1.1)
1. `job_start` launches **Smoke Stage** (`1/5/10` epochs or equivalent tiny budget).
2. Watcher completes; callback runs `SupervisorThenTask`.
3. Supervisor validates smoke:
   - exit code
   - required smoke artifacts
   - basic metric/log sanity (non-empty logs, parseable summary)
4. If smoke fails: run bounded debug loop on smoke only.
5. If smoke passes: launch **Full Stage** with normal budget.
6. On full-stage completion, supervisor classifies:
   - `PASS`
   - `RETRYABLE_FAILURE`
   - `CODE_BUG_OR_CONFIG_BUG`
   - `SUSPICIOUS_SUCCESS`
7. Supervisor takes bounded action and writes final explicit status.

## 5) Required Launch Contract (Default)
Every long run must include:
- `run_id`, `conversation_key`, `project_root`
- `watch.everySec`, `watch.tailLines`, `watch.thenTask`, `watch.runTasks=true`
- `smoke_cmd`, `full_cmd`
- `smoke_require_files[]` and `full_require_files[]`
- `smoke_budget_spec` (epochs/steps/rounds override)
- `cleanup_smoke_policy` in `{keep_all, keep_manifest_only}`

Recommended artifact expectations:
- smoke:
  - `<smoke_run_dir>/run.log`
  - `<smoke_run_dir>/meta.json`
  - `<smoke_run_dir>/summary.json`
- full:
  - `<full_run_dir>/run.log`
  - `<full_run_dir>/meta.json`
  - `<full_run_dir>/summary.json`
  - `<full_run_dir>/continuation_metrics.csv` (if continuation stage expected)

## 6) Stage 0 Smoke Policy
- Smoke budget should execute train + eval + checkpoint + summary paths, not train-only.
- Smoke should complete within 2-10 minutes where possible.
- Smoke defaults:
  - epochs in `{1, 5, 10}` or equivalent step budget
  - reduced data/rounds/horizons allowed
  - **do not change core algorithm path**
- Cleanup policy:
  - default `keep_manifest_only`: keep `summary.json`, error snippet, command, commit, and timestamps
  - optional deletion of bulk artifacts only after metadata persistence

## 7) Supervisor Policy (Bounded Loop)
### Budgets
- `max_infra_retries = 2`
- `max_debug_iterations = 3`
- `max_total_cycles = 6` (includes smoke debug + full debug)

### Terminal statuses
- `success`
- `blocked_infra_retry_exhausted`
- `blocked_debug_retry_exhausted`
- `blocked_repeated_signature`
- `blocked_human_decision_required`
- `blocked_policy_guardrail`

### Recovery classes
`RETRYABLE_FAILURE` (same command, no code edits):
- transient GPU wait/availability errors
- intermittent filesystem/network I/O
- non-deterministic transient OOM

`CODE_BUG_OR_CONFIG_BUG` (bounded debug iteration):
- deterministic Python stack traces
- argument/schema mismatch
- repeated deterministic signature on relaunch

`SUSPICIOUS_SUCCESS` (diagnostic iteration):
- exit code 0 but required artifact missing
- malformed/empty metrics artifact
- metric schema drift

### Stop conditions
Stop immediately and mark blocked when:
- same deterministic signature repeats after one patch attempt
- debug budget exceeded
- required fix is not mechanical (research/product decision)
- fix violates auto-edit guardrails

## 8) Iterative Debug Procedure (One Iteration)
1. Capture evidence:
   - error block from log
   - `run_id`, `job_id`, command, commit, dirty state
2. Build error signature:
   - exception type + top frame + normalized message hash
3. Apply minimal mechanical fix (if policy allows).
4. Run quick verification (`py_compile` / import / tiny execution path).
5. Relaunch only the failed stage (`smoke` or `full`) with incremented suffix.
6. Reclassify and update counters/state.

## 9) Auto-Fix Guardrails (Required)
Allowed:
- local, minimal fixes in declared allowlist paths
- argument wiring, import paths, null checks, bounds guards

Blocked (requires human decision):
- dependency upgrades/downgrades
- architecture or algorithm changes
- touching files outside allowlist
- disabling assertions/tests/safety checks

## 10) State Model
State path must be independent of per-attempt run dirs:
- `<project_root>/runs/supervisor/<conversation_key>/<run_id>/supervisor_state.json`

Suggested schema:
```json
{
  "run_id": "overnight_gap_hypothesis_20260224-xxxxxx",
  "conversation_key": "discord:...:thread:...",
  "status": "running|success|blocked_*",
  "phase": "smoke|full",
  "cycle": 3,
  "infra_retries_used": 1,
  "debug_iterations_used": 1,
  "last_error_signature": "IndexError@eval_pair:hash",
  "smoke_require_files": ["summary.json"],
  "full_require_files": ["summary.json", "continuation_metrics.csv"],
  "history": [
    {
      "cycle": 1,
      "phase": "smoke",
      "classification": "CODE_BUG_OR_CONFIG_BUG",
      "action": "patch+relaunch",
      "run_dir": "..."
    }
  ]
}
```

## 11) Risks and Caveats
1. Smoke false negatives:
- some bugs only appear at long horizon/late epochs.
- mitigation: keep full-stage supervisor loop active after smoke pass.

2. Smoke/full drift:
- if smoke config diverges too much, pass signal is weak.
- mitigation: enforce smoke overrides only on budget knobs.

3. Over-deletion of smoke artifacts:
- deleting all artifacts can erase audit trail.
- mitigation: keep manifest + summary + error excerpt by default.

4. Unsafe auto-patch behavior:
- broad edits can hide regressions.
- mitigation: strict edit allowlist + repeated-signature stop.

## 12) Acceptance Tests (Runtime-First)
1. Smoke gate pass path:
- smoke succeeds with required smoke files.
- expect: full stage launches exactly once.

2. Smoke gate fail path:
- inject deterministic smoke failure.
- expect: full stage is never launched.

3. Full-stage deterministic bug:
- smoke passes, full fails with known stack trace.
- expect: bounded debug iteration, repeated signature -> blocked.

4. Suspicious success:
- full exits 0 but required full artifact missing.
- expect: classified `SUSPICIOUS_SUCCESS`, final explicit blocked status if unresolved.

5. Budget enforcement:
- force repeated retryable failures.
- expect: retry caps respected, deterministic terminal status.

6. Policy guardrail:
- force a fix that requires dependency change.
- expect: immediate `blocked_policy_guardrail` without edit attempt.

## 13) Rollout Plan
Phase 1 (canary, one thread):
- enforce smoke gate + bounded supervisor for this thread only.
- track incident counts and false positive/false negative rates.

Phase 2 (research-wide):
- enable for all long-running experiment threads.
- retain strict budgets and explicit blocked outcomes.

Phase 3 (default):
- reject long-run submissions lacking smoke/full contract fields.

## 14) Minimal Implementation Checklist
1. Add long-run schema validator for smoke/full contract fields.
2. Add reusable `stage0_smoke_gate` runner.
3. Add supervisor state persistence under stable path.
4. Add required-artifact checker for both phases.
5. Add signature classifier + retry/debug budget gates.
6. Add policy guardrail evaluator for auto-fix decisions.
7. Add standard callback report formatter (phase, status, next action).

## 15) Recommended Initial Defaults
- `watch.everySec=300`
- `watch.tailLines=40`
- `smoke_budget=1-10 epochs` (or step-equivalent)
- `max_infra_retries=2`
- `max_debug_iterations=3`
- explicit `smoke_require_files` and `full_require_files` per experiment family

## 16) Reference Runner Template
Use `scripts/stage0_smoke_gate.py` as v1 execution shim:
```bash
python3 scripts/stage0_smoke_gate.py \
  --run-id <run_id> \
  --state-file runs/supervisor/<conversation_key>/<run_id>/state.json \
  --project-root /root/ebm-online-rl-prototype \
  --cwd /root/ebm-online-rl-prototype \
  --smoke-cmd "<experiment cmd with smoke budget>" \
  --full-cmd "<experiment cmd with full budget>" \
  --smoke-required-file "<smoke_run_dir>/summary.json" \
  --full-required-file "<full_run_dir>/summary.json" \
  --cleanup-smoke-policy keep_manifest_only
```
