import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.formula import Formula
from ase.io import read
from mattersim.forcefield.potential import MatterSimCalculator, Potential, build_dataloader
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm.auto import tqdm

matcher = StructureMatcher(
    ltol=0.4,
    stol=0.4,
    angle_tol=8,
    primitive_cell=True,
    scale=True,
    attempt_supercell=True,
    allow_subset=False,
    comparator=None,
    supercell_size="num_sites",
    ignored_species=[],
)


def dedup_samples(relaxed_all):
    relaxed_struct_list = [AseAtomsAdaptor.get_structure(struct) for struct in relaxed_all]

    dedup_sample_list = []
    deduped_num_list = []

    for sampled_structure in tqdm(relaxed_struct_list):
        has_match = False
        for idx, exisits_structure in enumerate(dedup_sample_list):
            if matcher.fit(sampled_structure, exisits_structure):
                has_match = True
                deduped_num_list[idx] += 1
                break

        if not has_match:
            dedup_sample_list.append(sampled_structure)
            deduped_num_list.append(1)

    print(
        f"Deduplication finished: {len(relaxed_struct_list)} "
        f"-> {len(dedup_sample_list)} samples after dedup"
    )

    return dedup_sample_list, deduped_num_list


def get_reduced_composition(atoms):
    """Get the reduced chemical formula from an ASE Atoms object."""
    reduced_composition = Formula(atoms.get_chemical_formula()).reduce()[0].format("hill")
    return reduced_composition


def read_atoms_from_path(sample_path, keyword="relaxed"):
    """Read all cif files from a given path."""
    file_list = glob.glob(f"{sample_path}/{keyword}*.cif")
    atoms_list = []
    for cif_file in file_list:
        try:
            atoms = read(cif_file)
        except Exception as e:
            print(f"cif_file {cif_file} read failed {e}")
            continue
        atoms_list.append(atoms)
    return atoms_list


def calc_energy_per_atom(atoms_list, calculator: MatterSimCalculator) -> np.ndarray:
    if not atoms_list:
        return np.array([])

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
    return energy_array / num_atoms


"""
usage: python sample_eval_utils/dedup_samples.py \
    --sample_path ./periodic_structures/rss_result/${composition}/converged/ \
    --target_path ./periodic_structures/rss_deduped/${composition}/
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--potential_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    calculator = MatterSimCalculator(load_path=args.potential_path, device=args.device)
    relaxed_file_list = list(Path(args.sample_path).glob("*.xyz"))
    relaxed_all, relaxed_formula_dict = [], defaultdict(list)

    for file in relaxed_file_list:
        atoms_list = read(file, index=":")
        relaxed_all.extend(atoms_list)
        for atoms in atoms_list:
            relaxed_formula_dict[atoms.get_chemical_formula()].append(atoms)

    if not relaxed_all:
        print("No structures found; skip dedup and dump.")
        raise SystemExit(0)

    os.makedirs(args.target_path, exist_ok=True)
    print(f"Read generated samples: {len(relaxed_all)} samples, {len(relaxed_file_list)} files")
    print(f"Sampled structure formula: {list(relaxed_formula_dict.keys())}")

    sample_energy = calc_energy_per_atom(relaxed_all, calculator)
    energy_min = float(np.min(sample_energy))
    print(f"Minimum energy (eV/atom) from raw samples: {energy_min:.6f}")

    dedupted_samples, deduped_nums = dedup_samples(relaxed_all)

    dedup_atoms = [AseAtomsAdaptor.get_atoms(struct) for struct in dedupted_samples]
    dedup_energy = calc_energy_per_atom(dedup_atoms, calculator)

    for idx, (atoms_to_save, num, energy) in enumerate(
        zip(dedup_atoms, deduped_nums, dedup_energy, strict=False)
    ):
        rel_energy = float(energy - energy_min)
        out_name = f"deduped_{idx}_num{num}_relE{rel_energy:.6f}.cif"
        atoms_to_save.write(Path(args.target_path) / out_name, format="cif")

    print("All deduped structures dumped.")
