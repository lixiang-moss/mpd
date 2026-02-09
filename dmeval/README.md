# DMEval 

DMEval is a lightweight, serial evaluation pipeline for benchmarking diffusion samplers (and other motion
planning “planners”) in a two-stage workflow:

- **Stage I (tune):** search over candidate patches for each sampler on a single scenario, then select best/top‑k.
- **Stage II (compare):** apply the Stage‑I best patches and run fair comparisons across multiple scenarios.



## Quick start

### Smoke test (no MPD required)

Run the full Stage I → Stage II loop with the built-in dummy planner (fast sanity check):

```bash
PYTHONPATH=dmeval/src python -m dmeval --config dmeval/conf/config_dummy_small.yaml run
```

This uses `dmeval/scripts/dummy_planner.py`, which writes `trial_metrics.jsonl` in the expected format so the
adapter/aggregation/ranking/plotting pipeline can be exercised without heavy external dependencies.



###  install into your environment

```bash
python -m pip install -e dmeval
# then:
dmeval run
```

If editable install fails in your environment, `dmeval/setup.py` exists as a minimal fallback for legacy toolchains.

## CLI overview

The CLI supports four subcommands:

- `tune` — Stage I only
- `compare` — Stage II only (requires Stage I outputs)
- `run` — Stage I then Stage II in one process (**recommended**)
- `explain` — print the resolved config after Hydra composition

Common flags:

- `--config <path>`: path to a Hydra YAML config (defaults to `dmeval/conf/config.yaml`).
- `-o key=value`: Hydra override (repeatable), e.g. `-o pipeline.root=outputs/exp1`.

Examples:

```bash
# Print resolved config (useful for debugging defaults/overrides)
PYTHONPATH=dmeval/src python -m dmeval --config dmeval/conf/config.yaml explain

# Run end-to-end with overrides
PYTHONPATH=dmeval/src python -m dmeval --config dmeval/conf/config.yaml \
  -o pipeline.root=outputs/my_run \
  -o pipeline.allow_overwrite=true \
  -o common_inference_args.device=cpu \
  run
```

## Configuration model (Hydra)

The main config (`dmeval/conf/config.yaml`) composes a few modular components:

- `planner`: how to invoke the system-under-test as a subprocess (e.g. `dmeval/conf/planner/mpd.yaml`)
- `adapter`: how to parse the planner’s result directory (e.g. `dmeval/conf/adapter/mpd.yaml`)
- `objective`: Stage‑I selection logic (constraints + ranking) (e.g. `dmeval/conf/objective/lexicographic.yaml`)

Recommended workflow:

1. Copy the default config, e.g. `cp dmeval/conf/config.yaml dmeval/conf/config_mpd.yaml`
2. Edit your experiment settings in the copy (scenario list, seeds, search space, device, etc.)
3. Keep `pipeline.root` stable if you run Stage I and Stage II separately
