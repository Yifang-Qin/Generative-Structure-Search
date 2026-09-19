# GSS: Generative Structure Search

GSS combines diffusion generative models with energy-based guidance from machine-learning force fields (MLFFs) to search for stable crystal structures. By injecting physical gradients into the diffusion sampling process, GSS achieves both broad structural coverage and efficient low-energy sampling -- overcoming the efficiency-coverage trade-off between conventional random structure search (RSS) and pure diffusion generation.

> **Note**: This repository covers the crystal structure search experiments. The molecular case study on aspirin reported in the paper lives in a separate repository: [Yifang-Qin/GSS-mol](https://github.com/Yifang-Qin/GSS-mol).

## Overview

Structure search aims to find stable configurations (local minima on the potential energy surface) for a given chemical composition. Two existing paradigms each have limitations:

- **Random Structure Search (RSS)**: generates random seed structures and relaxes them via force minimization. High coverage of the energy landscape, but low efficiency -- many samples land in high-energy basins.
- **Diffusion generative models**: learn the statistical distribution of realistic structures and generate samples efficiently. But they tend to concentrate around the global minimum and lack physical guarantees for unseen systems.

**GSS** combines both: at each diffusion sampling step, the denoising update is augmented with an energy gradient from an MLFF, biasing intermediate structures toward low-energy regions while the diffusion prior maintains structural diversity.

## Method

At each sampling step *i*, the structure is updated as:

```
M_{i+1} = M_i + alpha_i * f_diff(M_i) + beta_i * f_RSS(M_i)
```

where `f_diff` is the learned diffusion update and `f_RSS = -h * grad(E)` is the energy-based relaxation force. The coefficients `alpha_i` and `beta_i` follow complementary schedules: the diffusion prior dominates early steps for global structural guidance, while the energy term takes over in later steps to refine structures toward nearby local minima.

Three guidance strategies are implemented:
- **Mi** (M_i-guided): direct force/stress computation on the current noisy state M_i
- **MN** (M_N-guided): guidance through the predicted clean structure E[M_N|i]
- **M_tau**: guidance through an intermediate denoised state (experimental)

## Installation

Requires Python 3.10+ and a CUDA GPU. The project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

### Prerequisites

GSS depends on the following model weights (none are bundled with this repository):

| Directory | Contents | Source |
|-----------|----------|--------|
| `pretrained_mattersim/` | MatterSim force field checkpoint | [microsoft/mattersim](https://github.com/microsoft/mattersim/tree/main/pretrained_models) |
| `outputs/csp_model/checkpoints/` | Trained CSP diffusion model | Zenodo (see below) |
| `pretrained_uma/` | UMA force field checkpoint and references (optional) | [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) |

**MatterSim**: GSS uses the 5M-parameter variant by default. Download `mattersim-v1.0.0-5M.pth` from the [MatterSim `pretrained_models/` directory](https://github.com/microsoft/mattersim/tree/main/pretrained_models) and place it at `pretrained_mattersim/mattersim-v1.0.0-5M.pth`. The smaller `mattersim-v1.0.0-1M.pth` checkpoint can be substituted by overriding the `--matsim-path` CLI argument.

**UMA** can be obtained from the [FAIRChem repository](https://github.com/facebookresearch/fairchem) (optional, for alternative force field backend).

**CSP model**: the pretrained diffusion checkpoint (~460 MB) is hosted on Zenodo (DOI: [10.5281/zenodo.19651875](https://doi.org/10.5281/zenodo.19651875), which always resolves to the latest version). Download and place it as follows:

```bash
wget -O outputs/csp_model/checkpoints/last.ckpt \
  "https://zenodo.org/records/22847747/files/last.ckpt?download=1"
```

Alternatively, download manually from the [Zenodo record](https://doi.org/10.5281/zenodo.19651875) and place the file at `outputs/csp_model/checkpoints/last.ckpt`. The Zenodo record also ships `config.yaml`, which is identical to the one already tracked in this repository; no separate download is needed.

**AIRSS**: `gss-propose` requires the [AIRSS](https://www.mtg.msm.cam.ac.uk/Codes/AIRSS) toolkit. Install via [conda](https://github.com/conda-forge/airss-feedstock) or build from [source](https://airss-docs.github.io/getting-started/installation/). Not needed if you skip RSS proposal generation.

**Training from scratch** (optional): a training script (`scripts/train_csp_model.sh`) is provided for reference. Using the Zenodo checkpoint is strongly recommended. The training pipeline expects the alex-mp-20 dataset cached under `datasets/cache/alex_mp_20/`; for data preparation and training details, refer to the [MatterGen documentation](https://github.com/microsoft/mattergen).

## Directory Layout

All intermediate and output files are organized under `exp_results/{composition}/`:

```
exp_results/{comp}/
  rss_proposal/          # gss-propose output
  sampled/{config}/      # gss-search output
  relaxed/{config}/      # gss-relax converged structures
  unconverged/{config}/  # gss-relax unconverged structures
  deduped/               # gss-dedup output (reference set)
```

The `{config}` label is `rss` for the reference preparation pipeline, or the guidance method name (e.g., `mi`, `mn`) for the sampling pipeline. All composition and config values are case-insensitive.

## CLI Usage

After installation, five commands are available. For full parameter reference, see [CLI-REFERENCE.md](CLI-REFERENCE.md). The shell scripts under [`scripts/`](scripts/) are equivalent thin wrappers around the CLI for cluster jobs.

### `gss-propose` -- Generate RSS proposals

```bash
gss-propose --composition Al2O3 --multiplier 2 --num 1024
```

Generates random structure proposals for a given composition using AIRSS. The `--multiplier` flag creates supercells (e.g., Al2O3 x2 = Al4O6).

### `gss-search` -- Energy-guided diffusion generation

```bash
gss-search --composition al2o3 --guidance-method Mi
gss-search --composition al2o3 --guidance-method MN --pos-guide-coef 0.5
gss-search --composition al2o3 --guidance-method diffusion  # no guidance (baseline)
```

Runs the core GSS method: diffusion sampling with energy guidance from the MLFF. Requires a trained CSP model and a force field checkpoint.

### `gss-relax` -- Post-relaxation

```bash
gss-relax --composition al2o3 --config rss     # relax RSS proposals
gss-relax --composition al2o3 --config mi      # relax guided samples
```

Relaxes generated structures to local energy minima using MatterSim or UMA. Converged structures go to `relaxed/{config}/`, unconverged to `unconverged/{config}/`.

### `gss-dedup` -- Structure deduplication

```bash
gss-dedup --composition al2o3
```

Removes duplicate structures from `relaxed/rss/` using pymatgen's `StructureMatcher`. Used to build the reference set for evaluation.

### `gss-evaluate` -- Evaluation

```bash
gss-evaluate --composition al2o3 --config mi
```

Computes three metrics against the reference set (`deduped/`):
- **Coverage**: fraction of reference structures matched by generated samples
- **Efficiency**: average potential energy and low-energy ratio of samples
- **Budget Cost**: number of samples needed to cover the reference set

## Full Pipeline Example

```bash
# --- 1. Build reference set (RSS baseline) ---
gss-propose --composition Al2O3 --num 1024
gss-relax --composition al2o3 --config rss
gss-dedup --composition al2o3

# --- 2. Guided sampling ---
gss-search --composition al2o3 --guidance-method Mi
gss-relax --composition al2o3 --config mi

# --- 3. Evaluate ---
gss-evaluate --composition al2o3 --config mi
```

## Acknowledgments

The diffusion model backbone is built on [MatterGen](https://github.com/microsoft/mattergen) (Microsoft, MIT License). Energy guidance uses [MatterSim](https://github.com/microsoft/mattersim) and optionally [UMA](https://github.com/facebookresearch/fairchem) as force field backends.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
