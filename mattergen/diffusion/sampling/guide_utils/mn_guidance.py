import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.stress import voigt_6_to_full_3x3_stress
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.potential import Potential
from torch_geometric.data import Batch

from mattergen.diffusion.sampling.atoms_filters import DummyBatchCalculator, FrechetCellFilter


def obtain_diff_coeffs(
    batch,
    t,
    dt,
    pos_batch_idx,
    cell_batch_idx,
    pos_predictor,
    cell_predictor,
    tau=None,
):
    """Obtain the diffusion coefficients for the position and cell.
    Returns in dict form, all named following EDM's convention."""

    s = t + dt
    # note: pos_alpha_t remains 1 for VESDE across all timesteps
    pos_alpha_t, pos_sigma_t = pos_predictor.corruption.mean_coeff_and_std(
        x=batch.pos, t=t, batch_idx=pos_batch_idx, batch=batch
    )
    cell_alpha_t, cell_sigma_t = cell_predictor.corruption.mean_coeff_and_std(
        x=batch.cell, t=t, batch_idx=cell_batch_idx, batch=batch
    )
    pos_alpha_s, pos_sigma_s = pos_predictor.corruption.mean_coeff_and_std(
        x=batch.pos, t=s, batch_idx=pos_batch_idx, batch=batch
    )
    cell_alpha_s, cell_sigma_s = cell_predictor.corruption.mean_coeff_and_std(
        x=batch.cell, t=s, batch_idx=cell_batch_idx, batch=batch
    )
    if tau is not None:
        pos_alpha_tau, pos_sigma_tau = pos_predictor.corruption.mean_coeff_and_std(
            x=batch.pos, t=tau, batch_idx=pos_batch_idx, batch=batch
        )
        cell_alpha_tau, cell_sigma_tau = cell_predictor.corruption.mean_coeff_and_std(
            x=batch.cell, t=tau, batch_idx=cell_batch_idx, batch=batch
        )
        pos_x_coeff_tau = pos_alpha_tau / pos_alpha_t
        cell_x_coeff_tau = cell_alpha_tau / cell_alpha_t
        pos_score_coeff_tau = (pos_x_coeff_tau * (pos_sigma_tau**2)) - (pos_sigma_tau * pos_sigma_s)
        cell_score_coeff_tau = (cell_x_coeff_tau * (cell_sigma_tau**2)) - (
            cell_sigma_tau * cell_sigma_s
        )

    pos_x_coeff = pos_alpha_s / pos_alpha_t
    pos_score_coeff = (pos_x_coeff * (pos_sigma_t**2)) - (pos_sigma_t * pos_sigma_s)
    pos_std = torch.sqrt(1 - pos_alpha_t / pos_alpha_s)
    cell_x_coeff = cell_alpha_s / cell_alpha_t
    cell_score_coeff = (cell_x_coeff * (cell_sigma_t**2)) - (cell_sigma_t * cell_sigma_s)
    cell_std = torch.sqrt(1 - cell_alpha_t / cell_alpha_s)
    ret_dict = {
        "pos": {
            "alpha_t": pos_alpha_t,
            "sigma_t": pos_sigma_t,
            "alpha_s": pos_alpha_s,
            "sigma_s": pos_sigma_s,
            "x_coeff": pos_x_coeff,
            "score_coeff": pos_score_coeff,
            "std": pos_std,
        },
        "cell": {
            "alpha_t": cell_alpha_t,
            "sigma_t": cell_sigma_t,
            "alpha_s": cell_alpha_s,
            "sigma_s": cell_sigma_s,
            "x_coeff": cell_x_coeff,
            "score_coeff": cell_score_coeff,
            "std": cell_std,
        },
    }
    if tau is not None:
        ret_dict["pos"]["alpha_tau"] = pos_alpha_tau
        ret_dict["cell"]["alpha_tau"] = cell_alpha_tau
        ret_dict["pos"]["x_coeff_tau"] = pos_x_coeff_tau
        ret_dict["cell"]["x_coeff_tau"] = cell_x_coeff_tau
        ret_dict["pos"]["score_coeff_tau"] = pos_score_coeff_tau
        ret_dict["cell"]["score_coeff_tau"] = cell_score_coeff_tau

    return ret_dict


def tilde_f(
    batch,
    potential,
    pos_sigma_t,
    cell_sigma_t,
    gamma=0.0,
    num_samples=3,
    pos_step=1e-5,
    cell_step=1e-2,
):
    if gamma < 1e-5:
        f_sample, c_sample, no_guide_mask = get_energy_score(batch, potential)
        if no_guide_mask.any():
            pos_mask = torch.repeat_interleave(no_guide_mask, batch.num_atoms)
            f_sample[pos_mask] = torch.zeros_like(f_sample)[pos_mask]
            c_sample[no_guide_mask] = torch.zeros_like(c_sample)[no_guide_mask]
        return f_sample, c_sample

    pos_guidance_list = []
    cell_guidance_list = []
    no_guide_mask_list = []
    no_guide_pos_mask_list = []
    orig_pos, orig_cell = batch.pred_pos_tau.clone(), batch.pred_cell_tau.clone()
    for _ in range(num_samples):
        pos_pertub = orig_pos + pos_sigma_t * torch.randn_like(orig_pos) * pos_step * gamma
        cell_pertub = orig_cell + cell_sigma_t * torch.randn_like(orig_cell) * cell_step * gamma
        batch_sample = batch.replace(pos_to_guide=pos_pertub, cell_to_guide=cell_pertub)

        f_sample, c_sample, no_guide_mask = get_energy_score(batch_sample, potential)
        pos_mask = torch.repeat_interleave(no_guide_mask, batch.num_atoms)
        if no_guide_mask.any():
            f_sample[pos_mask] = torch.zeros_like(f_sample)[pos_mask]
            c_sample[no_guide_mask] = torch.zeros_like(c_sample)[no_guide_mask]

        pos_guidance_list.append(f_sample)
        cell_guidance_list.append(c_sample)
        no_guide_mask_list.append(no_guide_mask)
        no_guide_pos_mask_list.append(pos_mask)

    pos_guidance_list = torch.stack(pos_guidance_list, dim=0)
    cell_guidance_list = torch.stack(cell_guidance_list, dim=0)
    no_guide_mask_list = torch.stack(no_guide_mask_list, dim=0)
    no_guide_pos_mask_list = torch.stack(no_guide_pos_mask_list, dim=0)

    valid_guide_num = (~no_guide_mask_list).sum(dim=0).clamp(min=1)
    valid_pos_guide_num = torch.repeat_interleave(valid_guide_num, batch.num_atoms)
    pos_guidance_mean = pos_guidance_list.sum(dim=0) / valid_pos_guide_num[..., None]
    cell_guidance_mean = cell_guidance_list.sum(dim=0) / valid_guide_num[..., None, None]
    return pos_guidance_mean, cell_guidance_mean


def get_energy_score(
    batch: Batch, potential: Potential
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atoms_list: list[Atoms] = []
    filters_list: list[FrechetCellFilter] = []
    valid_indices = []
    data_list = batch.to_data_list()
    pos_to_guide_list = torch.split(batch.pos_to_guide, batch.num_atoms.cpu().tolist())

    no_guide_mask = torch.zeros(len(data_list), dtype=torch.bool, device=batch.pos.device)
    distance_threshold = 0.5
    for i, data in enumerate(data_list):
        fractional_pos = pos_to_guide_list[i].detach().cpu().numpy()
        fractional_pos = np.mod(fractional_pos, 1.0)  # ensure atoms stay inside the reference cell
        cell = batch.cell_to_guide[i].squeeze(0).detach().cpu().numpy()
        atomic_numbers = data.atomic_numbers.detach().cpu().numpy()
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
