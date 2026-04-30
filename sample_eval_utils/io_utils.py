"""Shared IO utilities for reading structures and computing per-atom energies."""

from pathlib import Path

import numpy as np
from ase.io import read
from mattersim.forcefield.potential import MatterSimCalculator, Potential, build_dataloader
from pymatgen.io.ase import AseAtomsAdaptor


def read_xyz_with_energy(
    xyz_files: list[Path], calculator: MatterSimCalculator
) -> tuple[list[dict], np.ndarray]:
    """Read .xyz files and compute per-atom energy for each structure."""
    structures_with_energy, atoms_list = [], []
    for file in xyz_files:
        atoms_list.extend(read(file, index=":"))

    potential: Potential = calculator.potential
    dataloader = build_dataloader(
        atoms_list,
        only_inference=True,
        model_type="m3gnet",
        batch_size=256,
    )
    energy_list, _, _ = potential.predict_properties(
        dataloader, include_forces=False, include_stresses=False
    )
    energy_array = np.array(energy_list)
    num_atoms = np.array([len(item) for item in atoms_list])
    energy_per_atom = energy_array / num_atoms

    for atoms, energy in zip(atoms_list, energy_per_atom, strict=True):
        structures_with_energy.append({
            "atoms": atoms,
            "energy": energy,
            "pymatgen_struct": AseAtomsAdaptor.get_structure(atoms),
        })
    return structures_with_energy, energy_per_atom


def read_cif_with_energy(
    cif_files: list[Path], calculator: MatterSimCalculator
) -> tuple[list[dict], np.ndarray]:
    """Read .cif files and compute per-atom energy for each structure."""
    structures_with_energy = []
    for cif_file in cif_files:
        atoms = read(cif_file)
        structures_with_energy.append({
            "atoms": atoms,
            "original_file": cif_file,
            "pymatgen_struct": AseAtomsAdaptor.get_structure(atoms),
        })

    atoms_list = [item["atoms"] for item in structures_with_energy]
    dataloader = build_dataloader(
        atoms_list,
        only_inference=True,
        model_type="m3gnet",
        batch_size=256,
    )
    energy_list, _, _ = calculator.potential.predict_properties(
        dataloader, include_forces=False, include_stresses=False
    )
    energy_array = np.array(energy_list)
    num_atoms = np.array([len(item["atoms"]) for item in structures_with_energy])
    energy_per_atom = energy_array / num_atoms

    for item, energy in zip(structures_with_energy, energy_per_atom, strict=True):
        item["energy"] = energy

    return structures_with_energy, energy_per_atom
