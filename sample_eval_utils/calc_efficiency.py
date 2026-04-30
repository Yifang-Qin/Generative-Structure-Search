"""
This script calculates the average potential energy and the ratio of low-energy
structures from a given set of CIF files using a deep learning potential.
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message=r"^Call to deprecated function.*")

import numpy as np
from mattersim.forcefield.potential import MatterSimCalculator

from sample_eval_utils.io_utils import read_xyz_with_energy


def calc_ratio_low_energy_structures(structures_with_energy, energy_threshold):
    """Calculate the ratio of low-energy structures."""
    if not structures_with_energy:
        return 0.0

    low_energy_count = sum(
        1 for item in structures_with_energy if item["energy"] < energy_threshold
    )
    ratio = low_energy_count / len(structures_with_energy)
    return ratio


if __name__ == "__main__":
    # --- Configuration ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential_path", type=str, required=True)
    parser.add_argument("--sample_dir", type=str, required=True)
    args = parser.parse_args()

    calculator = MatterSimCalculator(load_path=args.potential_path, device="cuda")
    sampled_files = list(Path(args.sample_dir).glob("*.xyz"))

    # Read structures and calculate energies
    sample_structures, sample_energy = read_xyz_with_energy(sampled_files, calculator)

    energy_min = np.min(sample_energy)
    energy_max = np.max(sample_energy)
    energy_threshold = min(energy_min + 0.1, energy_max)
    energy_max = min(energy_min + 1.0, energy_max)

    sampled_average_energy = np.mean(sample_energy)

    # Calculate average potential energy
    print(
        f"Sampled average potential energy (< {energy_max} eV/atom):"
        f" {sampled_average_energy:.4f} eV/atom"
    )

    # Calculate the ratio of low-energy structures
    sampled_ratio_low_energy = calc_ratio_low_energy_structures(sample_structures, energy_threshold)

    print(
        f"Sampled ratio of low-energy structures (< {energy_threshold} eV/atom):"
        f" {sampled_ratio_low_energy:.4f}"
    )
