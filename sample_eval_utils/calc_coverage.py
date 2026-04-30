"""
This script calculates the coverage of a set of sampled structures against a
set of reference structures. It bins structures by energy and checks for matches
using StructureMatcher.
"""

import argparse
import gc
import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message=r"^Call to deprecated function.*")


import numpy as np
import torch
from mattersim.forcefield.potential import MatterSimCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm.auto import tqdm

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


_SAMPLE_STRUCTS = None
_MATCHER = None


def _init_pool(sample_structs, matcher_dict):
    """Initializer for worker processes to avoid per-task large argument pickling."""
    global _SAMPLE_STRUCTS, _MATCHER
    _SAMPLE_STRUCTS = sample_structs
    _MATCHER = StructureMatcher.from_dict(matcher_dict)


def _has_match_for_ref(ref_struct):
    """Worker function: return True if any sample matches the ref_struct."""
    for sample_struct in _SAMPLE_STRUCTS:
        if _MATCHER.fit(ref_struct, sample_struct):
            return True
    return False


def _chunk_has_match_for_ref(args):
    """Worker function: check a chunk [start_idx:end_idx) of samples against ref_struct."""
    ref_struct, start_idx, end_idx = args
    for idx in range(start_idx, end_idx):
        if _MATCHER.fit(ref_struct, _SAMPLE_STRUCTS[idx]):
            return True
    return False


def bin_structures_by_energy(structures_with_energy, energy_granularity, valid_max_energy=100):
    """
    Bins structures by energy and filters out structures with energy above valid_max_energy.
    """
    structures_with_energy = [
        item for item in structures_with_energy if item["energy"] < valid_max_energy
    ]

    energy_bins = defaultdict(list)
    for item in structures_with_energy:
        bin_key_float = np.floor(item["energy"] / energy_granularity) * energy_granularity
        bin_key_str = f"{bin_key_float:.{max(0, -int(np.log10(energy_granularity)))}f}"
        energy_bins[bin_key_str].append(item)

    return energy_bins


def calculate_coverage(sampled_bins, reference_bins, num_workers: int = 0):
    """Calculates coverage of reference structures by sampled structures."""
    total_reference_count = 0
    matched_reference_count = 0
    coverage_info = []

    for bin_key, ref_structures in reference_bins.items():
        total_reference_count += len(ref_structures)

        if bin_key not in sampled_bins:
            continue

        sample_pymatgen_structs = [item["pymatgen_struct"] for item in sampled_bins[bin_key]]

        # Determine effective workers: 1=serial, 0=auto parallel, N>=2 explicit parallel
        effective_workers = num_workers if num_workers is not None else 1
        if effective_workers == 0:
            effective_workers = max(os.cpu_count() // 16 or 1, 1)

        # Serial path if single worker
        if effective_workers == 1:
            for ref_item in tqdm(ref_structures, desc="Calculating coverage", leave=False):
                coverage_info.append({
                    "energy": ref_item["energy"],
                    "structure": ref_item["pymatgen_struct"],
                    "matched": False,
                })
                ref_struct = ref_item["pymatgen_struct"]
                has_match = False
                for sample_struct in tqdm(
                    sample_pymatgen_structs, desc="Checking matches", leave=False
                ):
                    if matcher.fit(ref_struct, sample_struct):
                        has_match = True
                        coverage_info[-1]["matched"] = True
                        break

                if has_match:
                    matched_reference_count += 1
        else:
            # Parallel path: for each reference, parallelize over sampled structures in chunks
            # Outer loop is serial over references to enable early-stop per reference
            chunk_size = max(len(sample_pymatgen_structs) // (effective_workers * 8 or 1), 1)
            with ProcessPoolExecutor(
                max_workers=effective_workers,
                initializer=_init_pool,
                initargs=(sample_pymatgen_structs, matcher.as_dict()),
            ) as executor:
                for ref_item in tqdm(ref_structures, desc="Calculating coverage", leave=False):
                    coverage_info.append({
                        "energy": ref_item["energy"],
                        "structure": ref_item["pymatgen_struct"],
                        "matched": False,
                    })
                    coverage_idx = len(coverage_info) - 1
                    ref_struct = ref_item["pymatgen_struct"]

                    # Build chunk tasks
                    tasks = []
                    n = len(sample_pymatgen_structs)
                    for start in range(0, n, chunk_size):
                        end = min(start + chunk_size, n)
                        tasks.append((ref_struct, start, end))

                    has_match = False
                    for res in tqdm(
                        executor.map(_chunk_has_match_for_ref, tasks),
                        total=len(tasks),
                        desc="Checking matches (parallel)",
                        leave=False,
                    ):
                        if res:
                            has_match = True
                            break

                    if has_match:
                        coverage_info[coverage_idx]["matched"] = True
                        matched_reference_count += 1

    if total_reference_count > 0:
        coverage_rate = matched_reference_count / total_reference_count
        return coverage_rate, matched_reference_count, total_reference_count, coverage_info
    else:
        return 0, 0, 0, []


if __name__ == "__main__":
    # --- Configuration ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential_path", type=str, required=True)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--bin_size", type=float, default=10.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0=auto, 1=serial)",
    )
    args = parser.parse_args()

    calculator = MatterSimCalculator(load_path=args.potential_path, device="cuda")
    sampled_files = Path(args.sample_dir).glob("*.xyz")
    reference_files = Path(args.reference_dir).glob("*.cif")

    if not sampled_files or not reference_files:
        print("No CIF files found in one or both directories. Please check the paths.")
        exit(1)

    sampled_structures, sample_energy = read_xyz_with_energy(sampled_files, calculator)
    ref_structures, ref_energy = read_cif_with_energy(reference_files, calculator)

    print(f"Number of sampled: {len(sampled_structures)}, reference: {len(ref_structures)}")

    # Free CUDA memory as GPU is no longer needed
    del calculator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    energy_min = min(np.min(sample_energy), np.min(ref_energy))
    energy_max = max(np.max(sample_energy), np.max(ref_energy))
    energy_max = min(energy_min + 0.5, energy_max)

    sampled_mask = (sample_energy >= energy_min) & (sample_energy < energy_max + 0.2)
    reference_mask = (ref_energy >= energy_min) & (ref_energy < energy_max)

    sampled_structures = [
        struct for struct, mask in zip(sampled_structures, sampled_mask, strict=True) if mask
    ]
    ref_structures = [
        struct for struct, mask in zip(ref_structures, reference_mask, strict=True) if mask
    ]
    print(f"Filtered samples: {len(sampled_structures)}, reference: {len(ref_structures)}")

    sampled_bins = bin_structures_by_energy(sampled_structures, args.bin_size, energy_max)
    reference_bins = bin_structures_by_energy(ref_structures, args.bin_size, energy_max)
    print(f"Binned sampled structures into {len(sampled_bins)} energy bins.")
    print(f"Binned reference structures into {len(reference_bins)} energy bins.")

    print("\n--- Calculating Coverage ---")
    coverage_rate, matched_count, total_count, coverage_info = calculate_coverage(
        sampled_bins, reference_bins, num_workers=args.workers
    )

    print("\n--- Results ---")
    print(
        f"Coverage rate: {coverage_rate:.4f} ({matched_count}/{total_count} = {coverage_rate:.2%})"
    )
