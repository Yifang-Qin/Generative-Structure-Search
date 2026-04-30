import argparse
import os
import sys
import warnings
from glob import glob
from os.path import join

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message=r"^Call to deprecated function.*")

from ase import Atoms
from ase.io import read, write

sys.path.append(".")

from rss_utils.batch_relaxer import BatchRelaxer


def read_xyz_file(dir_path):
    file_list = glob(join(dir_path, "*.xyz")) + glob(join(dir_path, "*.extxyz"))
    atoms_list = []
    for file in file_list:
        atoms = read(file, index=":")
        atoms_list += atoms
    return atoms_list


def main(args):
    raw_atoms_list: list[Atoms] = read_xyz_file(args.source_dir)
    if len(raw_atoms_list) == 0:
        raise ValueError(f"No atoms found in {args.source_dir}")

    assert args.method.lower() in ["fire", "bfgs"], f"Invalid optimizer: {args.method}"

    if args.potential_backend == "matsim":
        from mattersim.forcefield.potential import Potential

        mlff = Potential.load(
            load_path=args.model_path,
            model_name=args.model_type,
        ).cuda()
        relaxer = BatchRelaxer(
            potential=mlff,
            optimizer=args.method.upper(),
            filter="EXPCELLFILTER",
            fmax=args.fmax,
            max_natoms_per_batch=1024,
            max_relaxation_step=args.num_steps,
        )
    elif args.potential_backend == "uma":
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from omegaconf import OmegaConf

        from rss_utils.batch_relaxer import BatchRelaxerUMA

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r".*The 'repr' attribute .* `Field\(\)` function.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r".*The 'frozen' attribute .* `Field\(\)` function.*",
        )

        checkpoint_path = join(args.uma_model_path, "checkpoints", args.uma_checkpoint)
        atom_refs_path = join(args.uma_model_path, "references", args.uma_atom_refs)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"UMA checkpoint not found: {checkpoint_path}")
        if not os.path.exists(atom_refs_path):
            raise FileNotFoundError(f"UMA atom refs not found: {atom_refs_path}")

        atom_refs = OmegaConf.load(atom_refs_path)
        predictor = load_predict_unit(
            checkpoint_path,
            inference_settings="default",
            overrides=None,
            device="cuda",
            atom_refs=atom_refs,
            workers=1,
        )
        uma_calculator = FAIRChemCalculator(predictor, task_name="omat")
        print(f"UMA loaded from {checkpoint_path}.")

        relaxer = BatchRelaxerUMA(
            potential=uma_calculator,
            optimizer=args.method.upper(),
            filter="FRECHETCELLFILTER",
            fmax=args.fmax,
            max_natoms_per_batch=4096,
            max_relaxation_step=args.num_steps,
        )

    converge_dict, unconverge_dict = relaxer.relax(raw_atoms_list)

    for key, atoms_list in converge_dict.items():
        # Clean calculator from atoms objects before writing to avoid array shape mismatch
        for atoms in atoms_list:
            atoms.calc = None
        write(
            join(args.output_dir, f"{key}.xyz"),
            atoms_list,
        )

    for key, atoms_list in unconverge_dict.items():
        # Clean calculator from atoms objects before writing to avoid array shape mismatch
        for atoms in atoms_list:
            atoms.calc = None
        write(
            join(args.unconverged_dir, f"{key}.xyz"),
            atoms_list,
        )
    converged_num = sum(len(atoms_list) for atoms_list in converge_dict.values())
    print(
        f"Relaxation finished, converged: {converged_num} / {len(raw_atoms_list)}\n"
        f"Convergence ratio: {converged_num / len(raw_atoms_list) * 100:.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--unconverged_dir", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)

    parser.add_argument("--fmax", type=float, default=0.001)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--method", type=str, default="fire")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="m3gnet")
    parser.add_argument(
        "--potential_backend",
        type=str,
        default="matsim",
        choices=["matsim", "uma"],
        help="Force field backend: matsim (MatterSim) or uma (UMA)",
    )
    parser.add_argument(
        "--uma_model_path",
        type=str,
        default=None,
        help="UMA model directory (required if potential_backend=uma)",
    )
    parser.add_argument(
        "--uma_checkpoint",
        type=str,
        default="uma-s-1p1.pt",
        help="UMA checkpoint filename",
    )
    parser.add_argument(
        "--uma_atom_refs",
        type=str,
        default="iso_atom_elem_refs.yaml",
        help="UMA atom reference energy filename",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.unconverged_dir, exist_ok=True)

    main(args)
