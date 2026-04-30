from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.stress import full_3x3_to_voigt_6_stress
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.datasets import data_list_collater
from fairchem.core.units.mlip_unit import load_predict_unit
from omegaconf import OmegaConf


class UMAPotential:
    """Adapter that aligns UMA FAIRChem models with MatterSim's Potential API."""

    def __init__(
        self,
        *,
        calculator: FAIRChemCalculator,
        device: str = "cuda",
    ) -> None:
        self._calculator = calculator
        self.predictor = calculator.predictor
        self.device = device

    def cuda(self) -> UMAPotential:
        """Return self for API compatibility; UMA runs on the configured device."""
        return self

    def to(self, device: str) -> UMAPotential:
        if device != self.device:
            raise ValueError(
                f"UMA predictor was initialized on {self.device}, "
                f"explicit device transfers are not supported."
            )
        return self

    def predict_properties(
        self,
        structures: Sequence[Atoms] | Iterable[Atoms] | Any,
        *,
        include_forces: bool = True,
        include_stresses: bool = True,
        **_: Any,
    ):
        """
        Compute (energy, forces, stress) for a batch of ASE structures.

        Args:
            structures: Either an iterable of ASE Atoms or a dataloader annotated with
                an `ase_atoms` attribute (see guidance utilities).
            include_forces: Whether to return forces.
            include_stresses: Whether to return stresses (Voigt-6 in GPa units).
        """

        atoms_list = self._extract_atoms(structures)
        if not atoms_list:
            outputs: list[Any] = [[]]
            if include_forces:
                outputs.append([])
            if include_stresses:
                outputs.append([])
            return tuple(outputs)

        atomic_graphs = [self._calculator.a2g(at) for at in atoms_list]
        batch = data_list_collater(atomic_graphs, otf_graph=True)
        predictions = self.predictor.predict(batch)

        energies = np.atleast_1d(predictions["energy"].detach().cpu().numpy()).reshape(-1)
        energies_list = [float(e) for e in energies]

        outputs: list[Any] = [energies_list]

        if include_forces:
            raw_forces = predictions["forces"].detach().cpu().numpy()
            natoms_list = [len(at) for at in atoms_list]
            idx = 0
            forces: list[np.ndarray] = []
            for natoms in natoms_list:
                forces.append(np.array(raw_forces[idx : idx + natoms], copy=True))
                idx += natoms
            outputs.append(forces)

        if include_stresses:
            raw_stress = predictions["stress"].detach().cpu().numpy()
            stresses: list[np.ndarray] = []
            for s in raw_stress:
                stress = np.asarray(s).squeeze()
                if stress.ndim == 1:
                    if stress.size == 6:
                        stress_voigt = stress
                    elif stress.size == 9:
                        stress_voigt = full_3x3_to_voigt_6_stress(stress.reshape(3, 3))
                    else:
                        raise ValueError(f"Unexpected stress shape: {stress.shape}")
                elif stress.ndim == 2:
                    if stress.shape == (3, 3):
                        stress_voigt = full_3x3_to_voigt_6_stress(stress)
                    elif stress.shape in {(1, 6), (6, 1)}:
                        stress_voigt = stress.reshape(6)
                    else:
                        raise ValueError(f"Unexpected stress shape: {stress.shape}")
                else:
                    raise ValueError(f"Unexpected stress ndim: {stress.ndim}")
                stresses.append(np.array(stress_voigt / units.GPa, copy=True))
            outputs.append(stresses)

        return tuple(outputs)

    def _extract_atoms(self, structures: Sequence[Atoms] | Iterable[Atoms] | Any) -> list[Atoms]:
        if isinstance(structures, Sequence):
            if not structures:
                return []
            if all(isinstance(item, Atoms) for item in structures):
                return [at.copy() for at in structures]

        ase_atoms = getattr(structures, "ase_atoms", None)
        if ase_atoms is not None:
            if isinstance(ase_atoms, Sequence):
                return [at.copy() for at in ase_atoms]
            raise TypeError("`ase_atoms` attribute must be a sequence of ASE Atoms.")

        raise TypeError(
            "UMA potential expects a sequence of ASE Atoms or a dataloader "
            "annotated with `ase_atoms`."
        )


def load_uma_potential(
    *,
    model_path: str,
    checkpoint_filename: str = "uma-m-1p1.pt",
    atom_refs_filename: str = "iso_atom_elem_refs.yaml",
    device: str = "cuda",
    inference_settings: str = "default",
    overrides: dict | None = None,
    workers: int = 1,
    task_name: str = "omat",
) -> UMAPotential:
    """Load UMA weights and return a Potential-compatible adapter."""

    checkpoint_path = _resolve_subpath(model_path, "checkpoints", checkpoint_filename)
    atom_refs_path = _resolve_subpath(model_path, "references", atom_refs_filename)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"UMA checkpoint not found: {checkpoint_path}")
    if not os.path.exists(atom_refs_path):
        raise FileNotFoundError(f"UMA atom refs not found: {atom_refs_path}")

    atom_refs_conf = OmegaConf.load(atom_refs_path)
    atom_refs = OmegaConf.to_container(atom_refs_conf, resolve=True)

    predictor = load_predict_unit(
        checkpoint_path,
        inference_settings=inference_settings,
        overrides=overrides,
        device=device,
        atom_refs=atom_refs,
    )
    calculator = FAIRChemCalculator(predict_unit=predictor, task_name=task_name)
    return UMAPotential(calculator=calculator, device=device)


def _resolve_subpath(base: str, subdir: str, maybe_path: str) -> str:
    if os.path.isabs(maybe_path) and os.path.exists(maybe_path):
        return maybe_path
    candidate = os.path.join(base, subdir, maybe_path)
    return candidate
