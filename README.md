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

## System requirements

- **Operating system**: Linux x86_64. Tested on Ubuntu 22.04.5 LTS. The pinned PyTorch and PyG wheels are Linux builds; macOS and Windows are not supported.
- **Python**: 3.10 (`pyproject.toml` pins `== 3.10.*`).
- **GPU**: an NVIDIA GPU with a driver that supports CUDA 12.4 (driver >= 550). The CUDA 12.4 runtime ships inside the PyTorch wheel (`torch==2.6.0+cu124`), so no separate CUDA toolkit is needed. Tested on NVIDIA H20 (96 GB, driver 580.95). The quick demo below peaks at about 9 GB of GPU memory; the default `gss-search` batch of 512 structures needs more.
- **Disk and RAM**: about 6.6 GB for the virtual environment plus 0.55 GB of model weights; the quick demo uses less than 2.5 GB of host RAM.
- **Software dependencies**: exact versions are pinned in `pyproject.toml` and `uv.lock`. The main ones are torch 2.6.0+cu124, torch-geometric 2.7 (2.8.1 at the time of testing), torch-scatter 2.1.2 / torch-sparse 0.6.18 / torch-cluster 1.6.3 (`pt26cu124` wheels from data.pyg.org), pytorch-lightning 2.0.6, hydra-core 1.3.1, numpy 2.2.6, ase 3.27.0, pymatgen 2025.10.7 and mattersim 1.2.1.
- **Non-standard hardware**: none beyond the CUDA GPU.

## Installation

Requires Python 3.10 and a CUDA GPU. The project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Typical install time: 5-10 min on a workstation with a 100 Mbps connection. About 3 GB of wheels are downloaded (PyTorch, the CUDA runtime libraries and Triton account for most of it) and the environment takes 6.6 GB on disk. On our test server (Ubuntu 22.04, fast PyPI mirror) the whole sequence took 3 min: `pip install uv` 1 min, `uv venv` a few seconds, `uv pip install -e .` 2 min. If `uv` is not found after `pip install uv`, add `~/.local/bin` to your `PATH` or call it as `python -m uv`.

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

**AIRSS**: `gss-propose` requires the [AIRSS](https://www.mtg.msm.cam.ac.uk/Codes/AIRSS) toolkit. Install via [conda](https://github.com/conda-forge/airss-feedstock) or build from [source](https://airss-docs.github.io/getting-started/installation/). Not needed for the [quick demo](#quick-demo-no-airss-required) below, which writes random seed structures with ASE instead.

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

Runs the core GSS method: diffusion sampling with energy guidance from the MLFF. Requires a trained CSP model, a force field checkpoint, and seed structures in `exp_results/{comp}/rss_proposal/*.xyz` (written by `gss-propose`, or by the snippet in the quick demo below); each diffusion trajectory starts from one seed structure.

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

## Quick demo (no AIRSS required)

Searches for Si structures with 64 guided samples and evaluates them against the Si reference set shipped on Zenodo. Run everything from the repository root with `.venv` activated and the two checkpoints from [Prerequisites](#prerequisites) in place.

**1. Seed structures.** `gss-search` starts each diffusion trajectory from a seed structure read from `exp_results/{comp}/rss_proposal/*.xyz`, normally written by `gss-propose` (AIRSS). For the demo, write 64 random 8-atom Si cells with ASE instead:

```bash
mkdir -p exp_results/si/rss_proposal
python - <<'EOF'
import numpy as np
from ase import Atoms
from ase.io import write

rng = np.random.default_rng(0)
seeds = []
while len(seeds) < 64:
    vol = 8 * 20.0 * rng.uniform(0.8, 1.2)  # ~20 A^3 per Si atom, +-20 %
    cell = np.eye(3) * vol ** (1 / 3) + rng.uniform(-0.5, 0.5, (3, 3))
    atoms = Atoms("Si8", scaled_positions=rng.random((8, 3)), cell=cell, pbc=True)
    if atoms.get_all_distances(mic=True)[np.triu_indices(8, 1)].min() > 1.5:  # min separation 1.5 A
        seeds.append(atoms)
write("exp_results/si/rss_proposal/si8.xyz", seeds)
EOF
```

**2. Reference set.** The Zenodo record ships `gss_reference_structures.tar.gz` (1.8 MB): the deduplicated RSS reference sets of the 12 systems in the paper, laid out as `<System>/<System>_0000.cif, ...` plus an `index.csv`. `gss-evaluate` reads every `*.cif` in `exp_results/{comp}/deduped/` and ignores file names, so copying the Si folder there replaces the `gss-propose` -> `gss-relax --config rss` -> `gss-dedup` steps of the full pipeline:

```bash
wget -O gss_reference_structures.tar.gz \
  "https://zenodo.org/records/22847747/files/gss_reference_structures.tar.gz?download=1"
tar -xzf gss_reference_structures.tar.gz
mkdir -p exp_results/si/deduped
cp gss_reference_structures/Si/*.cif exp_results/si/deduped/   # 281 CIFs
```

**3. Search, relax, evaluate.**

```bash
gss-search --composition si --guidance-method Mi --batch-size 64 --max-samples 64
gss-relax --composition si --config mi
gss-evaluate --composition si --config mi
```

**Expected output and run time** (measured on one NVIDIA H20; a different GPU changes the times, not the files):

| Step | Files written | Wall time | Peak GPU memory |
|------|---------------|-----------|-----------------|
| `gss-search` | `exp_results/si/sampled/mi/generated_crystals.extxyz` (64 structures) and `generated_crystals_cif.zip` (`gen_0.cif` ... `gen_63.cif`) | ~6.5 min | 4.5 GB |
| `gss-relax` | `exp_results/si/relaxed/mi/Si8.xyz` (converged structures) and `exp_results/si/unconverged/mi/Si8.xyz`; prints `Relaxation finished, converged: 61 / 64` | ~2.5 min | 3.2 GB |
| `gss-evaluate` | nothing on disk; prints the three metric blocks below | ~2 min | 9.3 GB |

Evaluation printout from the run above (abridged; diffusion sampling is stochastic, so expect the same order of magnitude rather than identical numbers):

```
=== Computing coverage ===
Number of sampled: 61, reference: 281
Filtered samples: 61, reference: 230
Coverage rate: 0.6913 (159/230 = 69.13%)

=== Computing energy efficiency ===
Sampled average potential energy (< -4.748 eV/atom): -5.1652 eV/atom
Sampled ratio of low-energy structures (< -5.309 eV/atom): 0.1311

=== Computing sample budget ===
Number of filtered reference structures: 7
Unable to reach reference coverage threshold, only 6 / 7 matched (target_count=7)
Sample budget to cover references in window (N_ref=7, target_ratio=1.000, avg over 10 runs): 0.00
```

With only 64 samples the budget metric usually reports `0.00` (not every low-energy reference is hit). The paper protocol uses the `gss-search` defaults (`--batch-size 512 --max-samples 1024`, i.e. 16x more samples) and AIRSS proposals from `gss-propose`.

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
