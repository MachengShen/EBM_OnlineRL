# Agentic Maze2D External-Proposal Implementation Notes

Last updated: 2026-02-17

## Purpose
This document contains implementation-level details for the external-proposal workflow added to the Maze2D agentic autodecider. It is intended as a technical reference so user-facing summaries can stay high-level.

## Files
- `scripts/agentic_maze2d_autodecider.py`
- `scripts/launch_agentic_maze2d_autodecider_tmux.sh`

## Added Functions
- `_coerce_trial_config(raw)`
  - Validates and converts a raw JSON trial object into `base.TrialConfig`.
- `_coerce_external_proposal(raw, idx)`
  - Validates one external proposal object and converts it into a `Proposal`.
- `_load_external_proposals(path, max_new)`
  - Loads proposal JSON from disk, supports list or `{ "proposals": [...] }`, truncates to `max_new`.
- `_summarize_recent_rows(rows, limit=8)`
  - Produces compact recent-run summaries for round context files.
- `_serialize_tried_keys(tried)`
  - Converts tried-config tracking keys into JSON-serializable form.
- `_build_round_context(...)`
  - Builds the round context payload written to `round_###_context.json`.
- `_wait_for_external_proposals(...)`
  - Writes context file, polls for `round_###_proposals.json`, returns loaded proposals or timeout status.

## New CLI Args
In `agentic_maze2d_autodecider.py`:
- `--proposal-source {internal,external,external_preferred}`
- `--proposal-dir`
- `--base-dir` (optional explicit output directory)
- `--proposal-wait-timeout-sec`
- `--proposal-poll-sec`

## Runtime Behavior
- `proposal-source=internal`
  - Uses existing heuristic/stochastic `_reasoned_proposals(...)` logic.
- `proposal-source=external`
  - Emits round context file and strictly waits for external proposal file.
  - Stops current agentic run if proposals are missing/empty after timeout.
- `proposal-source=external_preferred`
  - Tries external first; falls back to internal heuristics on missing/empty proposals.

## External File Contract
Per round (index `k`):
- Context output:
  - `<proposal_dir>/round_%03d_context.json`
- Proposal input:
  - `<proposal_dir>/round_%03d_proposals.json`

Accepted proposal input formats:
- Top-level list of proposal objects
- Object with key `proposals` containing list of proposal objects

Each proposal requires:
- `trial.train_steps`
- `trial.online_rounds`
- `trial.online_collect_episodes_per_round`
- `trial.online_train_steps_per_round`
- `trial.online_replan_every_n_steps`
- `trial.online_goal_geom_p`

Optional fields:
- `proposal_id`
- `rationale`
- `origin`
- `risk_level` (normalized to `low` or `med`)
- `common_overrides`

## Launcher Defaults
In `launch_agentic_maze2d_autodecider_tmux.sh`:
- `PROPOSAL_SOURCE` defaults to `external`
- `PROPOSAL_DIR` defaults to `runs/analysis/synth_maze2d_diffuser_probe/proposal_exchange_<session>`
- Optional explicit output root:
  - if `BASE_DIR` is set, the launcher forwards `--base-dir "${BASE_DIR}"`
- Wait/poll args are forwarded:
  - `--proposal-wait-timeout-sec`
  - `--proposal-poll-sec`

## Observability and Outputs
- `agentic_config.json` includes proposal-source configuration.
- `summary.json` includes `proposal_source` and `proposal_dir`.
- `agentic.log` includes external wait/load/timeout status lines.

## Validation Performed
- `python3 -m py_compile scripts/agentic_maze2d_autodecider.py`
- `bash -n scripts/launch_agentic_maze2d_autodecider_tmux.sh`
- Venv smoke tests for:
  - Successful external load path
  - Timeout path when no proposal file is written
