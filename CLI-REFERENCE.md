# GSS CLI Reference

Five commands are available after `pip install -e .`:

All paths are derived from `--composition` and `--config`. No manual path specification is needed.

| Command        | Description                                             |
| -------------- | ------------------------------------------------------- |
| `gss-propose`  | Generate random structure proposals via AIRSS           |
| `gss-search`   | Energy-guided diffusion structure search                |
| `gss-relax`    | Post-relax generated structures with a force field      |
| `gss-dedup`    | Deduplicate relaxed structures to build reference set   |
| `gss-evaluate` | Evaluate coverage, energy efficiency, and sample budget |


---

## `gss-propose`

Generate random structure proposals for a given composition using AIRSS.

```bash
gss-propose --composition Al2O3 --multiplier 2 --num 1024
```

Output: `exp_results/{comp}/rss_proposal/{formula}.xyz`

| Parameter       | Type | Default  | Description                                 |
| --------------- | ---- | -------- | ------------------------------------------- |
| `--composition` | str  | required | Chemical formula, e.g. `Al2O3`, `CaTiO3`   |
| `--multiplier`  | int  | `1`      | Supercell multiplier applied to atom counts |
| `--num`         | int  | `512`    | Number of random proposals to generate      |
| `--varvol`      | int  | `20`     | AIRSS `VARVOL` parameter (volume variation) |


---

## `gss-search`

Run energy-guided diffusion sampling for crystal structure search.

```bash
# Mi-guided (default, recommended)
gss-search --composition al2o3 --guidance-method Mi

# MN-guided
gss-search --composition al2o3 --guidance-method MN

# Unguided diffusion baseline
gss-search --composition al2o3 --guidance-method diffusion
```

Output: `exp_results/{comp}/sampled/{config}/`

| Parameter               | Type  | Default                                          | Description                                                                                   |
| ----------------------- | ----- | ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `--composition`         | str   | required                                         | Chemical formula, e.g. `al2o3` (case-insensitive)                                             |
| `--guidance-method`     | str   | `Mi`                                             | Guidance strategy: `Mi`, `MN`, `M_tau`, or `diffusion` (no guidance). Case-insensitive.       |
| `--config`              | str   | same as `guidance-method`                        | Run label used as subdirectory name                                                           |
| `--guidance-config`     | str   | `./sampling_conf/guidance.yaml`                  | Path to guidance config YAML                                                                  |
| `--pos-guide-coef`      | float | method default                                   | Strength of position guidance ($\beta_i$ scaling for atomic forces)                           |
| `--cell-guide-coef`     | float | method default                                   | Strength of cell guidance ($\beta_i$ scaling for stress)                                      |
| `--t-mid`               | int   | `750`                                            | Sigmoid schedule midpoint $t_\text{mid}$; controls when energy guidance dominates (Mi only)   |
| `--t-scale`             | int   | `250`                                            | Sigmoid schedule width $t_\text{scale}$ (Mi only)                                             |
| `--guidance-steps`      | int   | method default                                   | Number of trailing diffusion steps with guidance active (e.g. `750` = last 750 of 1000 steps) |
| `--batch-size`          | int   | `512`                                            | Structures generated per batch                                                                |
| `--max-samples`         | int   | `1024`                                           | Total structures to generate                                                                  |
| `--model-path`          | str   | `./outputs/csp_model/checkpoints/`               | CSP model checkpoint directory                                                                |
| `--potential-backend`   | str   | `matsim`                                         | Force field backend: `matsim` or `uma`                                                        |
| `--matsim-path`         | str   | `./pretrained_mattersim/mattersim-v1.0.0-5M.pth` | MatterSim checkpoint path                                                                     |
| `--record-trajectories` | bool  | `False`                                          | Save full diffusion trajectories                                                              |


**Method defaults** (from `sampling_conf/guidance.yaml`):


| Method  | `pos-guide-coef` | `cell-guide-coef` | `guidance-steps` |
| ------- | ---------------- | ----------------- | ---------------- |
| `Mi`    | `5e-5`           | `2e-1`            | `750`            |
| `MN`    | `1e-4`           | `1e-1`            | `400`            |
| `M_tau` | `1e-4`           | `1e-1`            | `400`            |


---

## `gss-relax`

Post-relax generated structures to local energy minima using MatterSim or UMA.

```bash
gss-relax --composition al2o3 --config rss     # relax RSS proposals
gss-relax --composition al2o3 --config mi      # relax guided samples
```

Input: `exp_results/{comp}/rss_proposal/` (when config is `rss`) or `exp_results/{comp}/sampled/{config}/` (otherwise)
Output: `exp_results/{comp}/relaxed/{config}/` (converged) and `exp_results/{comp}/unconverged/{config}/` (not converged)

| Parameter             | Type  | Default                                          | Description                                            |
| --------------------- | ----- | ------------------------------------------------ | ------------------------------------------------------ |
| `--composition`       | str   | required                                         | Chemical formula (case-insensitive)                    |
| `--config`            | str   | required                                         | Run label: `rss` for proposals, or guidance method name |
| `--potential-backend` | str   | `matsim`                                         | Force field backend: `matsim` or `uma`                 |
| `--matsim-path`       | str   | `./pretrained_mattersim/mattersim-v1.0.0-5M.pth` | MatterSim checkpoint path                              |
| `--fmax`              | float | `0.001`                                          | Force convergence threshold (eV/A)                     |
| `--max-steps`         | int   | `1000`                                           | Maximum relaxation steps per structure                 |
| `--method`            | str   | `fire`                                           | Optimizer: `fire` or `bfgs`                            |


---

## `gss-dedup`

Remove duplicate relaxed RSS structures to build the reference set using pymatgen `StructureMatcher`.

```bash
gss-dedup --composition al2o3
```

Input: `exp_results/{comp}/relaxed/rss/`
Output: `exp_results/{comp}/deduped/`

| Parameter          | Type | Default                                          | Description                                 |
| ------------------ | ---- | ------------------------------------------------ | ------------------------------------------- |
| `--composition`    | str  | required                                         | Chemical formula (case-insensitive)         |
| `--potential-path` | str  | `./pretrained_mattersim/mattersim-v1.0.0-5M.pth` | MatterSim checkpoint for energy calculation |
| `--device`         | str  | `cuda`                                           | Device for energy calculation               |


---

## `gss-evaluate`

Evaluate structural coverage, energy efficiency, and sample budget cost against the reference set.

```bash
gss-evaluate --composition al2o3 --config mi
```

Sample dir: `exp_results/{comp}/relaxed/{config}/`
Reference dir: `exp_results/{comp}/deduped/`

The following matches the evaluation protocol in the paper (these are also the default values):

```bash
gss-evaluate --composition al2o3 --config mi \
    --energy-threshold 0.1 --budget-threshold 1.0 --bin-size 10.0
```


| Parameter            | Type  | Default                                          | Description                                                                                                                                           |
| -------------------- | ----- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--composition`      | str   | required                                         | Chemical formula (case-insensitive)                                                                                                                   |
| `--config`           | str   | required                                         | Run label matching the config used in `gss-search` / `gss-relax`                                                                                     |
| `--potential-path`   | str   | `./pretrained_mattersim/mattersim-v1.0.0-5M.pth` | MatterSim checkpoint for energy calculation                                                                                                           |
| `--energy-threshold` | float | `0.1`                                            | $E_\text{hull}$ threshold (eV/atom) for low-energy ratio and budget                                                                                   |
| `--budget-threshold` | float | `1.0`                                            | Fraction of reference structures to cover when computing budget cost                                                                                  |
| `--bin-size`         | float | `10.0`                                           | Energy bin width for coverage calculation (eV/atom); default exceeds the filter threshold so all samples fall in one bin, matching the paper protocol |
| `--workers`          | int   | `0`                                              | Parallel workers for coverage computation (`0` = auto)                                                                                                |
