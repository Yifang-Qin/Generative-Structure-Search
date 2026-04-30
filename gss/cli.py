"""GSS (Generative Structure Search) CLI entry points.

Each public function corresponds to a console_script entry point (gss-propose,
gss-search, gss-relax, gss-dedup, gss-evaluate).  All input/output paths are
derived from --composition and --config; no manual path selection is needed.

Directory layout under exp_results/:
  exp_results/{comp}/rss_proposal/      <- gss-propose output
  exp_results/{comp}/sampled/{config}/  <- gss-search output
  exp_results/{comp}/relaxed/{config}/  <- gss-relax converged output
  exp_results/{comp}/unconverged/{config}/ <- gss-relax unconverged output
  exp_results/{comp}/deduped/             <- gss-dedup output (reference set)
"""

import re
import warnings
from pathlib import Path

import fire

# Suppress pydantic warnings from upstream dependencies (hydra/omegaconf)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _multiply_formula(formula: str, multiplier: int) -> str:
    """Multiply atom counts in a chemical formula by a scalar.

    Example: _multiply_formula("Al2O3", 2) -> "Al4O6"
    """
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    result = ""
    for element, count_str in parts:
        if not element:
            continue
        count = int(count_str) if count_str else 1
        result += f"{element}{count * multiplier}"
    return result


def _exp_root(comp: str) -> Path:
    return Path(f"./exp_results/{comp}")


# ---------------------------------------------------------------------------
# gss-propose
# ---------------------------------------------------------------------------


def _propose(
    composition: str,
    multiplier: int = 1,
    num: int = 512,
    varvol: int = 20,
):
    """Generate RSS random structure proposals via AIRSS.

    Output: exp_results/{comp}/rss_proposal/{formula}.xyz

    Args:
        composition: Chemical formula (e.g., Al2O3, CaTiO3).
        multiplier: Supercell multiplier applied to atom counts.
        num: Number of structures to generate.
        varvol: AIRSS VARVOL parameter.
    """
    from rss_utils.gen_rss_comp import (
        collect_res_files_and_write_xyz,
        run_airss,
        write_rss_cell_file,
    )

    comp = composition.lower()
    formula = _multiply_formula(composition, multiplier)

    output_dir = _exp_root(comp) / "rss_proposal"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = str(output_dir / f"{formula.lower()}.xyz")

    write_rss_cell_file(formula, varvol, "rss.cell")
    run_airss(num)
    collect_res_files_and_write_xyz(formula, output)


def propose():
    fire.Fire(_propose)


# ---------------------------------------------------------------------------
# gss-search
# ---------------------------------------------------------------------------


def _search(
    composition: str,
    model_path: str = "./outputs/csp_model/checkpoints/",
    guidance_method: str = "Mi",
    config: str | None = None,
    guidance_config: str = "./sampling_conf/guidance.yaml",
    potential_backend: str = "matsim",
    matsim_path: str = "./pretrained_mattersim/mattersim-v1.0.0-5M.pth",
    batch_size: int = 512,
    max_samples: int = 1024,
    pos_guide_coef: float | None = None,
    cell_guide_coef: float | None = None,
    t_mid: int | None = None,
    t_scale: int | None = None,
    guidance_steps: int | None = None,
    uma_model_path: str | None = None,
    uma_checkpoint: str = "uma-s-1p1.pt",
    uma_atom_refs: str = "iso_atom_elem_refs.yaml",
    uma_device: str = "cuda",
    uma_inference_settings: str = "default",
    uma_workers: int = 1,
    checkpoint_epoch: str = "last",
    record_trajectories: bool = False,
):
    """Energy-guided diffusion crystal structure generation.

    Output: exp_results/{comp}/sampled/{config}/
    RSS proposals read from: exp_results/{comp}/rss_proposal/

    Args:
        composition: Chemical formula (e.g., Si, Al2O3).
        model_path: Model checkpoint directory.
        guidance_method: Guidance method: Mi, MN, M_tau, or diffusion (no guidance).
        config: Run label used as subdirectory name. Defaults to guidance_method.
        guidance_config: Guidance config YAML path.
        potential_backend: matsim or uma.
        matsim_path: MatterSim checkpoint path.
        batch_size: Batch size for generation.
        max_samples: Maximum number of structures to generate.
        checkpoint_epoch: Checkpoint epoch to load (best, last, or int).
        record_trajectories: Whether to save diffusion trajectories.
    """
    from mattergen.scripts.generate import main as generate_main

    comp = composition.lower()
    cfg = (config or guidance_method).lower()

    output = str(_exp_root(comp) / "sampled" / cfg)
    rss_proposal_path = str(_exp_root(comp) / "rss_proposal")
    Path(output).mkdir(parents=True, exist_ok=True)

    matsim_arg = matsim_path if potential_backend == "matsim" else None
    uma_arg = uma_model_path if potential_backend == "uma" else None

    generate_main(
        output_path=output,
        model_path=model_path,
        batch_size=batch_size,
        num_batches=1,
        checkpoint_epoch=checkpoint_epoch,
        sampling_config_name="csp",
        record_trajectories=record_trajectories,
        matsim_path=matsim_arg,
        uma_model_path=uma_arg,
        uma_checkpoint=uma_checkpoint,
        uma_atom_refs=uma_atom_refs,
        uma_device=uma_device,
        uma_inference_settings=uma_inference_settings,
        uma_workers=uma_workers,
        rss_proposal_path=rss_proposal_path,
        max_sampling_num=max_samples,
        guidance_config_path=guidance_config,
        guidance_method=guidance_method,
        pos_guide_coef=pos_guide_coef,
        cell_guide_coef=cell_guide_coef,
        t_mid=t_mid,
        t_scale=t_scale,
        guidance_steps=guidance_steps,
    )


def search():
    fire.Fire(_search)


# ---------------------------------------------------------------------------
# gss-relax
# ---------------------------------------------------------------------------


def _relax(
    composition: str,
    config: str,
    potential_backend: str = "matsim",
    matsim_path: str = "./pretrained_mattersim/mattersim-v1.0.0-5M.pth",
    fmax: float = 0.001,
    max_steps: int = 1000,
    method: str = "fire",
    uma_model_path: str | None = None,
    uma_checkpoint: str = "uma-s-1p1.pt",
    uma_atom_refs: str = "iso_atom_elem_refs.yaml",
):
    """Post-relaxation of structures using MatterSim or UMA force field.

    Input:  exp_results/{comp}/rss_proposal/       (when config == "rss")
            exp_results/{comp}/sampled/{config}/    (otherwise)
    Output: exp_results/{comp}/relaxed/{config}/    (converged)
            exp_results/{comp}/unconverged/{config}/ (not converged)

    Args:
        composition: Chemical formula (e.g., Si, Al2O3).
        config: Run label matching the config used in gss-propose / gss-search.
                Use "rss" to relax RSS proposals; otherwise matches gss-search config.
        potential_backend: matsim or uma.
        matsim_path: MatterSim checkpoint path.
        fmax: Force convergence threshold (eV/Angstrom).
        max_steps: Maximum relaxation steps.
        method: Optimizer: fire or bfgs.
    """
    import argparse
    import os

    from rss_utils.rss_relaxation import main as relax_main

    comp = composition.lower()
    cfg = config.lower()
    root = _exp_root(comp)

    if cfg == "rss":
        source_dir = str(root / "rss_proposal")
    else:
        source_dir = str(root / "sampled" / cfg)

    output_dir = str(root / "relaxed" / cfg)
    unconverged_dir = str(root / "unconverged" / cfg)
    log_path = str(root / "relaxed" / cfg / "dyn.log")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(unconverged_dir, exist_ok=True)

    args = argparse.Namespace(
        source_dir=source_dir,
        output_dir=output_dir,
        unconverged_dir=unconverged_dir,
        log_path=log_path,
        fmax=fmax,
        num_steps=max_steps,
        method=method,
        potential_backend=potential_backend,
        model_path=matsim_path,
        model_type="m3gnet",
        uma_model_path=uma_model_path,
        uma_checkpoint=uma_checkpoint,
        uma_atom_refs=uma_atom_refs,
    )
    relax_main(args)


def relax():
    fire.Fire(_relax)


# ---------------------------------------------------------------------------
# gss-dedup
# ---------------------------------------------------------------------------


def _dedup(
    composition: str,
    potential_path: str = "./pretrained_mattersim/mattersim-v1.0.0-5M.pth",
    device: str = "cuda",
):
    """Deduplicate relaxed RSS structures to build the reference set.

    Input:  exp_results/{comp}/relaxed/rss/
    Output: exp_results/{comp}/deduped/

    Args:
        composition: Chemical formula (e.g., Si, Al2O3).
        potential_path: MatterSim checkpoint for energy calculation.
        device: Device for energy calculation.
    """
    import os

    import numpy as np
    from ase.io import read
    from mattersim.forcefield.potential import MatterSimCalculator
    from pymatgen.io.ase import AseAtomsAdaptor

    from sample_eval_utils.dedup_samples import calc_energy_per_atom, dedup_samples

    comp = composition.lower()
    root = _exp_root(comp)

    sample_dir = root / "relaxed" / "rss"
    output_dir = root / "deduped"
    os.makedirs(output_dir, exist_ok=True)

    calculator = MatterSimCalculator(load_path=potential_path, device=device)

    relaxed_file_list = list(sample_dir.glob("*.xyz"))
    relaxed_all = []
    for file in relaxed_file_list:
        atoms_list = read(file, index=":")
        relaxed_all.extend(atoms_list)

    if not relaxed_all:
        print("No structures found; skip dedup.")
        return

    print(f"Read {len(relaxed_all)} samples from {len(relaxed_file_list)} files")

    sample_energy = calc_energy_per_atom(relaxed_all, calculator)
    energy_min = float(np.min(sample_energy))
    print(f"Minimum energy (eV/atom): {energy_min:.6f}")

    deduped_samples, deduped_nums = dedup_samples(relaxed_all)

    dedup_atoms = [AseAtomsAdaptor.get_atoms(struct) for struct in deduped_samples]
    dedup_energy = calc_energy_per_atom(dedup_atoms, calculator)

    for idx, (atoms_to_save, num, energy) in enumerate(
        zip(dedup_atoms, deduped_nums, dedup_energy, strict=False)
    ):
        rel_energy = float(energy - energy_min)
        out_name = f"deduped_{idx}_num{num}_relE{rel_energy:.6f}.cif"
        atoms_to_save.write(output_dir / out_name, format="cif")

    print(f"Saved {len(deduped_samples)} deduped structures to {output_dir}")


def dedup():
    fire.Fire(_dedup)


# ---------------------------------------------------------------------------
# gss-evaluate
# ---------------------------------------------------------------------------


def _evaluate(
    composition: str,
    config: str,
    potential_path: str = "./pretrained_mattersim/mattersim-v1.0.0-5M.pth",
    workers: int = 0,
    bin_size: float = 10.0,
    energy_threshold: float = 0.1,
    budget_threshold: float = 1.0,
):
    """Evaluate structures: coverage, energy efficiency, and sample budget.

    Sample dir:    exp_results/{comp}/relaxed/{config}/
    Reference dir: exp_results/{comp}/deduped/

    Args:
        composition: Chemical formula (e.g., Si, Al2O3).
        config: Run label matching the config used in gss-search / gss-relax.
        potential_path: MatterSim checkpoint path.
        workers: Parallel workers for coverage (0=auto).
        bin_size: Energy bin size for coverage calculation.
        energy_threshold: Threshold for low-energy ratio and budget.
        budget_threshold: Fraction of references to cover for budget.
    """
    import subprocess
    import sys

    comp = composition.lower()
    cfg = config.lower()
    root = _exp_root(comp)

    sample_dir = str(root / "relaxed" / cfg)
    reference_dir = str(root / "deduped")

    # 1. Coverage
    print("=== Computing coverage ===")
    subprocess.run(
        [
            sys.executable,
            "sample_eval_utils/calc_coverage.py",
            "--potential_path",
            potential_path,
            "--sample_dir",
            sample_dir,
            "--reference_dir",
            reference_dir,
            "--workers",
            str(workers),
            "--bin_size",
            str(bin_size),
        ],
        check=True,
    )

    # 2. Energy efficiency
    print("\n=== Computing energy efficiency ===")
    subprocess.run(
        [
            sys.executable,
            "sample_eval_utils/calc_efficiency.py",
            "--potential_path",
            potential_path,
            "--sample_dir",
            sample_dir,
        ],
        check=True,
    )

    # 3. Sample budget
    print("\n=== Computing sample budget ===")
    subprocess.run(
        [
            sys.executable,
            "sample_eval_utils/evaluate_efficiency.py",
            "--potential_path",
            potential_path,
            "--sample_dir",
            sample_dir,
            "--reference_dir",
            reference_dir,
            "--energy_threshold",
            str(energy_threshold),
            "--budget_threshold",
            str(budget_threshold),
        ],
        check=True,
    )

    print("\n=== Evaluation complete ===")


def evaluate():
    fire.Fire(_evaluate)
