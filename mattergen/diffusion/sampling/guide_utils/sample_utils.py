from collections.abc import Callable
from functools import partial
from typing import Literal, TypeVar

import torch

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling.guide_utils.guidance_conf import GuidanceConf
from mattergen.diffusion.sampling.guide_utils.mi_guidance import get_energy_score as mi_guidance
from mattergen.diffusion.sampling.guide_utils.mn_guidance import obtain_diff_coeffs, tilde_f
from mattergen.diffusion.sampling.pc_partials import PredictorPartial

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time


def mi_update(
    samples_means: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch: Diffusable,
    score: Diffusable,
    t_step: int,
    t_continuous: torch.Tensor,
    dt: torch.Tensor,
    guidance_conf: GuidanceConf,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    force_frac = torch.zeros_like(score.pos)
    cell_strain = torch.zeros_like(score.cell)
    pos_delta = samples_means["pos"][0] - batch.pos
    cell_delta = samples_means["cell"][0] - batch.cell

    force_frac, cell_strain, no_guide_mask = mi_guidance(batch, guidance_conf.potential)
    force_frac = force_frac * guidance_conf.pos_guide_coef
    cell_strain = cell_strain * guidance_conf.cell_guide_coef
    if no_guide_mask.any():
        pos_mask = torch.repeat_interleave(no_guide_mask, batch.num_atoms)
        force_frac[pos_mask] = score.pos[pos_mask]
        cell_strain[no_guide_mask] = score.cell[no_guide_mask]

    sigmoid_logit = torch.full(
        [1], (t_step - guidance_conf.t_mid) / guidance_conf.t_scale
    )
    relative_weight = torch.sigmoid(sigmoid_logit).to(pos_delta)
    pos_delta = (1 - relative_weight) * pos_delta + relative_weight * force_frac
    cell_delta = (1 - relative_weight) * cell_delta + relative_weight * cell_strain

    updated_pos = torch.remainder(batch.pos + pos_delta, 1.0)
    updated_cell = batch.cell + cell_delta
    samples_means["pos"] = (updated_pos, updated_pos)
    samples_means["cell"] = (updated_cell, updated_cell)
    return samples_means


def mn_update(
    samples_means: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch: Diffusable,
    score: Diffusable,
    t_step: int,
    t_continuous: torch.Tensor,
    dt: torch.Tensor,
    guidance_conf: GuidanceConf,
    multi_corruption: MultiCorruption,
    predictors: dict[str, PredictorPartial],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    batch = batch.replace(pos_to_guide=batch.pos.clone(), cell_to_guide=batch.cell.clone())
    tau_step = t_continuous * guidance_conf.tau
    edm_coef = obtain_diff_coeffs(
        batch,
        t_continuous,
        dt,
        multi_corruption._get_batch_indices(batch)["pos"],
        multi_corruption._get_batch_indices(batch)["cell"],
        predictors["pos"],
        predictors["cell"],
        tau=tau_step,
    )

    cell_r = batch.cell_r_mean
    cell_epsilon = -score.cell * edm_coef["cell"]["sigma_t"]
    pos_t, cell_t = samples_means["pos"][0].clone(), samples_means["cell"][0].clone()
    for _ in range(guidance_conf.n_recur):
        if guidance_conf.tau_guidance:
            pred_pos_tau = (
                edm_coef["pos"]["x_coeff_tau"] * pos_t
                + edm_coef["pos"]["score_coeff_tau"] * score.pos
            )
            pred_cell_tau_diff = (
                edm_coef["cell"]["x_coeff_tau"] * (cell_t - cell_r)
                + edm_coef["cell"]["score_coeff_tau"] * cell_epsilon
            )
            pred_cell_tau = cell_r + pred_cell_tau_diff
            batch = batch.replace(pred_pos_tau=pred_pos_tau, pred_cell_tau=pred_cell_tau)
        else:
            pred_pos_0 = (1 / edm_coef["pos"]["alpha_t"]) * (
                pos_t + (edm_coef["pos"]["sigma_t"] ** 2) * score.pos
            )
            pred_cell_diff = (1 / edm_coef["cell"]["alpha_t"]) * (
                (cell_t - cell_r) - (edm_coef["cell"]["sigma_t"] ** 2) * cell_epsilon
            )
            pred_cell_0 = cell_r + pred_cell_diff
            batch = batch.replace(pred_pos_tau=pred_pos_0, pred_cell_tau=pred_cell_0)
            edm_coef["pos"]["alpha_tau"] = torch.ones_like(edm_coef["pos"]["alpha_t"])
            edm_coef["cell"]["alpha_tau"] = torch.ones_like(edm_coef["cell"]["alpha_t"])

        delta_t_pos, delta_tau_pos, delta_t_cell, delta_tau_cell = (
            torch.zeros_like(pos_t),
            torch.zeros_like(pos_t),
            torch.zeros_like(cell_t),
            torch.zeros_like(cell_t),
        )
        if guidance_conf.tfg_rho > 0:
            raise NotImplementedError("Delta_t guidance requires calculation of Model Jacobian")
        if guidance_conf.tfg_mu > 0:
            for _ in range(guidance_conf.n_iter):
                force_frac, cell_strain = tilde_f(
                    batch,
                    guidance_conf.potential,
                    edm_coef["pos"]["sigma_t"],
                    edm_coef["cell"]["sigma_t"],
                    guidance_conf.tfg_gamma,
                )
                force_frac = force_frac * guidance_conf.pos_guide_coef
                cell_strain = cell_strain * guidance_conf.cell_guide_coef

                delta_tau_pos += guidance_conf.tfg_mu * force_frac
                delta_tau_cell += guidance_conf.tfg_mu * cell_strain
                batch = batch.replace(
                    pos_to_guide=batch.pred_pos_tau + delta_tau_pos,
                    cell_to_guide=batch.pred_cell_tau + delta_tau_cell,
                )

        pos_t_m_1 = (
            pos_t * edm_coef["pos"]["x_coeff"]
            + score.pos * edm_coef["pos"]["score_coeff"]
            + delta_t_pos * edm_coef["pos"]["x_coeff"]
            + (delta_tau_pos * edm_coef["pos"]["alpha_s"] / edm_coef["pos"]["alpha_tau"])
        )
        cell_t_m_1_diff = (
            (cell_t - cell_r) * edm_coef["cell"]["x_coeff"]
            + score.cell * edm_coef["cell"]["score_coeff"]
            + delta_t_cell * edm_coef["cell"]["x_coeff"]
            + (delta_tau_cell * edm_coef["cell"]["alpha_s"] / edm_coef["cell"]["alpha_tau"])
        )
        cell_t_m_1 = cell_t_m_1_diff + cell_r

        z_pos, z_cell = torch.randn_like(pos_t), torch.randn_like(cell_t)
        pos_t = torch.remainder(
            pos_t_m_1 / edm_coef["pos"]["x_coeff"] + edm_coef["pos"]["std"] * z_pos, 1.0
        )
        cell_t = (
            cell_t_m_1_diff / edm_coef["cell"]["x_coeff"] + edm_coef["cell"]["std"] * z_cell
        ) + cell_r

    pos_t_m_1 = torch.remainder(pos_t_m_1, 1.0)
    samples_means["pos"] = (pos_t_m_1, pos_t_m_1)
    samples_means["cell"] = (cell_t_m_1, cell_t_m_1)
    return samples_means


def no_guidance_update(
    samples_means: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch: Diffusable,
    score: Diffusable,
    t_step: int,
    t_continuous: torch.Tensor,
    dt: torch.Tensor,
    guidance_conf: GuidanceConf | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return samples_means


def guidance_factory(
    guidance_method: Literal["Mi", "MN", "M_tau"] | None,
    guidance_conf: GuidanceConf,
    multi_corruption: MultiCorruption,
    predictors: dict[str, PredictorPartial],
) -> Callable[
    [
        dict[str, tuple[torch.Tensor, torch.Tensor]],
        Diffusable,
        Diffusable,
        int,
        torch.Tensor,
        torch.Tensor,
        GuidanceConf | None,
    ],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]:
    if guidance_method is None:
        return no_guidance_update
    elif guidance_method == "Mi":
        return partial(mi_update, guidance_conf=guidance_conf)
    elif guidance_method in ["MN", "M_tau"]:
        return partial(
            mn_update,
            guidance_conf=guidance_conf,
            multi_corruption=multi_corruption,
            predictors=predictors,
        )

    raise ValueError(f"Invalid guidance method: {guidance_method}")
