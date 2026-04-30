import numpy as np
import torch
from ase import Atoms, units
from ase.neighborlist import NeighborList
from ase.stress import voigt_6_to_full_3x3_stress
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.potential import Potential
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch

from mattergen.diffusion.sampling.atoms_filters import DummyBatchCalculator, FrechetCellFilter


def get_energy_score(
    batch: Batch, potential: Potential
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atoms_list: list[Atoms] = []
    filters_list: list[FrechetCellFilter] = []
    valid_indices = []
    data_list = batch.to_data_list()

    no_guide_mask = torch.zeros(len(data_list), dtype=torch.bool, device=batch.pos.device)
    distance_threshold = 0.5
    for i, data in enumerate(data_list):
        fractional_pos = data.pos.detach().cpu().numpy()
        fractional_pos = np.mod(fractional_pos, 1.0)  # keep scaled positions inside the cell
        atomic_numbers = data.atomic_numbers.detach().cpu().numpy()
        cell = data.cell.squeeze(0).detach().cpu().numpy()
        orig_cell = batch.cell_r[i].detach().cpu().numpy()

        atoms = Atoms(
            scaled_positions=fractional_pos,
            numbers=atomic_numbers,
            cell=cell,
            pbc=[True, True, True],
        )
        atoms.calc = DummyBatchCalculator()

        no_guide_mask[i] = check_short_distances(atoms, distance_threshold)
        if not no_guide_mask[i]:
            atoms_list.append(atoms.copy())
            filters_list.append(FrechetCellFilter(atoms, orig_cell=orig_cell))
            valid_indices.append(i)

    pred_forces = [torch.zeros_like(data.pos) for data in data_list]
    pred_stresses = [torch.zeros(3, 3, device=batch.pos.device) for _ in data_list]

    if atoms_list:
        potential_loader = build_dataloader(
            atoms_list,
            only_inference=True,
            model_type="m3gnet",
        )
        potential_loader.ase_atoms = atoms_list
        with torch.enable_grad():
            predictions = potential.predict_properties(
                potential_loader, include_forces=True, include_stresses=True
            )

        for j, i in enumerate(valid_indices):
            filters_list[j].info["total_energy"] = predictions[0][j]
            filters_list[j].arrays["forces"] = predictions[1][j]
            filters_list[j].info["stress"] = predictions[2][j]

            pred_forces[i] = torch.from_numpy(
                filters_list[j].get_forces()[: len(atoms_list[j])]
            ).to(batch.pos)
            pred_forces[i] = pred_forces[i] @ batch.cell[i].T

            pred_stresses[i] = -torch.from_numpy(
                voigt_6_to_full_3x3_stress(filters_list[j].stress)
            ).to(batch.pos)  # notice that the stress is the **negative** of the gradient

    force_frac = torch.cat(pred_forces, dim=0)
    cell_step_batch = torch.stack(pred_stresses, dim=0)

    return force_frac, cell_step_batch, no_guide_mask


def get_energy_score_old(
    batch: Batch, potential: Potential
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forces, stresses, no_guide_mask = get_force_stress(batch, potential)

    cell_r, cell_r_inv = batch.cell_r, batch.cell_r_inv
    cell_t, cell_t_inv = batch.cell, torch.inverse(batch.cell)

    force_padded = pad_sequence(forces, batch_first=True, padding_value=0.0)
    stress_batch = torch.stack(stresses, dim=0)

    pos_list = [data.pos.squeeze(0) for data in batch.to_data_list()]
    pos_padded = pad_sequence(pos_list, batch_first=True, padding_value=0.0)

    pos_t = pos_padded @ cell_t
    num_atoms = batch.num_atoms
    cell_volumes = torch.det(cell_t)

    force_step_padded = (
        ((-cell_volumes) / (num_atoms**2))[..., None, None]
        * pos_t
        @ stress_batch
        @ cell_t_inv
        @ cell_r
        @ cell_r
        @ cell_t_inv
    ) + (force_padded @ cell_t @ cell_r_inv @ cell_r_inv @ cell_t)

    cell_step_batch = (
        (1 / (num_atoms**2))[..., None, None]
        * cell_r
        @ cell_r
        @ cell_t_inv
        @ (cell_volumes[..., None, None] * stress_batch)
    )
    cell_step_batch = 0.5 * (cell_step_batch + cell_step_batch.transpose(-2, -1))

    force_frac_step_padded = force_step_padded @ cell_t_inv
    force_frac_list = [force_frac_step_padded[i, :num_atom] for i, num_atom in enumerate(num_atoms)]
    force_frac = torch.cat(force_frac_list, dim=0)

    return force_frac, cell_step_batch, no_guide_mask


def check_short_distances(atoms: Atoms, threshold: float = 1.0) -> bool:
    """Return True if there is any atom with short distance to other atoms."""
    try:
        radii = [threshold / 2.0] * len(atoms)
        nl = NeighborList(radii, skin=0.0, self_interaction=False, bothways=False)
        nl.update(atoms)

        for i in range(len(atoms)):
            indices, _ = nl.get_neighbors(i)
            if len(indices) > 0:
                return True
        return False
    except Exception:
        return True


def get_force_stress(
    batch: Batch, potential: Potential
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    atoms_list = []
    valid_indices = []
    data_list = batch.to_data_list()

    no_guide_mask = torch.zeros(len(data_list), dtype=torch.bool, device=batch.pos.device)
    distance_threshold = 1.0
    for i, data in enumerate(data_list):
        fractional_pos = data.pos.detach().cpu().numpy()
        fractional_pos = np.mod(fractional_pos, 1.0)  # wrap into the unit cell
        atomic_numbers = data.atomic_numbers.detach().cpu().numpy()
        cell = data.cell.squeeze(0).detach().cpu().numpy()

        atoms = Atoms(
            scaled_positions=fractional_pos,
            numbers=atomic_numbers,
            cell=cell,
            pbc=[True, True, True],
        )

        no_guide_mask[i] = check_short_distances(atoms, distance_threshold)
        if not no_guide_mask[i]:
            atoms_list.append(atoms)
            valid_indices.append(i)

    pred_forces = [torch.zeros_like(data.pos) for data in data_list]
    pred_stresses = [torch.zeros(3, 3, device=batch.pos.device) for _ in data_list]

    if atoms_list:
        potential_loader = build_dataloader(
            atoms_list,
            only_inference=True,
            model_type="m3gnet",
        )
        potential_loader.ase_atoms = atoms_list
        with torch.enable_grad():
            predictions = potential.predict_properties(
                potential_loader, include_forces=True, include_stresses=True
            )

        for j, i in enumerate(valid_indices):
            pred_forces[i] = torch.from_numpy(predictions[1][j]).to(batch.pos)
            pred_stresses[i] = torch.from_numpy(predictions[2][j]).to(batch.pos) * units.GPa

    return pred_forces, pred_stresses, no_guide_mask
