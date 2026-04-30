"""
This script calculates the average potential energy and the ratio of low-energy
structures from a given set of CIF files using a deep learning potential.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message=r"^Call to deprecated function.*")

from mattersim.forcefield.potential import MatterSimCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher

from sample_eval_utils.io_utils import read_cif_with_energy, read_xyz_with_energy

matcher = StructureMatcher(
    ltol=0.5,
    stol=0.5,
    angle_tol=8,
    primitive_cell=True,
    scale=True,
    attempt_supercell=True,
    allow_subset=False,
    comparator=None,
    supercell_size="num_sites",
    ignored_species=[],
)


if __name__ == "__main__":
    # --- Configuration ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential_path", type=str, required=True)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--budget_threshold", type=float, default=None)
    parser.add_argument("--energy_threshold", type=float, default=0.1)
    args = parser.parse_args()

    # --- Args validation / defaults ---
    # budget_threshold is a ratio in (0, 1], indicating the fraction of filtered
    # reference structures that need to be covered for the budget calculation.
    # If not provided, default to 1.0 (cover all filtered references).
    if args.budget_threshold is None:
        args.budget_threshold = 1.0
    if not (0.0 < float(args.budget_threshold) <= 1.0):
        raise ValueError(f"--budget_threshold must be in (0, 1], got {args.budget_threshold}")

    calculator = MatterSimCalculator(load_path=args.potential_path, device="cuda")

    sampled_files = Path(args.sample_dir).glob("*.xyz")
    reference_files = Path(args.reference_dir).glob("*.cif")

    sampled_structures, sample_energy = read_xyz_with_energy(sampled_files, calculator)
    ref_structures, ref_energy = read_cif_with_energy(reference_files, calculator)

    energy_min = np.min(sample_energy)
    energy_max = np.max(sample_energy)
    energy_max = min(energy_min + 10.0, energy_max)

    # Use reference structure's energy range for filtering references
    # This ensures consistent filtering regardless of sample energy distribution
    ref_energy_min = np.min(ref_energy)
    ref_energy_max = np.max(ref_energy)
    ref_energy_threshold = min(ref_energy_min + args.energy_threshold, ref_energy_max)

    sampled_mask = (sample_energy >= energy_min) & (sample_energy < energy_max)
    sampled_structures_filtered = [
        struct for struct, mask in zip(sampled_structures, sampled_mask, strict=True) if mask
    ]

    print(f"#sampled structures within the energy window: {len(sampled_structures_filtered)}")
    print(f"#reference structures: {len(ref_structures)}")

    ref_mask = (ref_energy >= ref_energy_min) & (ref_energy < ref_energy_threshold)
    ref_structures_filtered = [
        struct for struct, mask in zip(ref_structures, ref_mask, strict=True) if mask
    ]

    sample_pmg = [item["pymatgen_struct"] for item in sampled_structures_filtered]
    ref_pmg = [item["pymatgen_struct"] for item in ref_structures_filtered]

    num_samples = len(sample_pmg)
    num_refs = len(ref_pmg)
    num_runs = 10

    avg_budget = None
    if num_samples == 0:
        print("No sampled structures after filtering; sample_budget_for_stable=0")
        avg_budget = 0.0
    elif num_refs == 0:
        print("No reference structures after filtering; sample_budget_for_stable=0")
        print("Sample budget to cover all references in window (N_ref=0): 0")
        avg_budget = 0.0
    else:
        print(f"Number of filtered reference structures: {num_refs}")
        target_ratio = float(args.budget_threshold)
        target_count = int(np.ceil(target_ratio * num_refs))
        target_count = max(1, min(target_count, num_refs))
        if target_ratio < 1.0:
            print(
                "Budget threshold enabled: need to cover at least "
                f"{target_ratio:.3f} of references "
                f"(target_count={target_count}/{num_refs})"
            )
        else:
            print("Budget threshold disabled (1.0): need to cover all references")

        rng = np.random.default_rng()
        budgets: list[float] = []
        for _ in range(num_runs):
            indices = np.arange(num_samples)
            rng.shuffle(indices)
            matched_refs_loop = np.zeros(num_refs, dtype=bool)
            budget_loop = None
            for seen_count, idx in enumerate(
                tqdm(indices, desc="Matching samples", leave=False), start=1
            ):
                s = sample_pmg[idx]
                # Check against all references; mark matches; stop when all matched
                for j, r in enumerate(ref_pmg):
                    if not matched_refs_loop[j] and matcher.fit(r, s):
                        matched_refs_loop[j] = True
                # Stop when covered enough references (ratio-based threshold)
                if int(matched_refs_loop.sum()) >= target_count:
                    budget_loop = seen_count
                    break
            if budget_loop is None:
                budget_loop = 0.0
                budgets.append(budget_loop)
                print(
                    "Unable to reach reference coverage threshold, "
                    f"only {matched_refs_loop.sum()} / {num_refs} matched "
                    f"(target_count={target_count})"
                )
                break
            budgets.append(budget_loop)

        avg_budget = float(np.mean(budgets)) if len(budgets) > 0 else 0.0
        print(
            "Sample budget to cover references in window "
            f"(N_ref={num_refs}, target_ratio={target_ratio:.3f}, "
            f"avg over {num_runs} runs): {avg_budget:.2f}"
        )
