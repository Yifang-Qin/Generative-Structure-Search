# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Mapping
from typing import Generic, TypeVar

import torch
from tqdm.auto import tqdm

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.sampling.guide_utils import GuidanceConf
from mattergen.diffusion.sampling.guide_utils.sample_utils import guidance_factory
from mattergen.diffusion.sampling.pc_partials import CorrectorPartial, PredictorPartial

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time
SampleAndMean = tuple[Diffusable, Diffusable]
SampleAndMeanAndMaybeRecords = tuple[Diffusable, Diffusable, list[Diffusable] | None]
SampleAndMeanAndRecords = tuple[Diffusable, Diffusable, list[Diffusable]]


class PredictorCorrector(Generic[Diffusable]):
    """Generates samples using predictor-corrector sampling."""

    def __init__(
        self,
        *,
        diffusion_module: DiffusionModule,
        predictor_partials: dict[str, PredictorPartial] | None = None,
        corrector_partials: dict[str, CorrectorPartial] | None = None,
        device: torch.device,
        n_steps_corrector: int,
        N: int,  # noqa: N803
        eps_t: float = 1e-3,
        max_t: float | None = None,
    ):
        self._diffusion_module = diffusion_module
        self.N = N

        if max_t is None:
            max_t = self._multi_corruption.T
        assert max_t <= self._multi_corruption.T, "Denoising cannot start from beyond T"

        self._max_t = max_t
        assert corrector_partials or predictor_partials, (
            "Must specify at least one predictor or corrector"
        )
        corrector_partials = corrector_partials or {}
        predictor_partials = predictor_partials or {}
        if self._multi_corruption.discrete_corruptions:
            # These all have property 'N' because they are D3PM type
            assert {c.N for c in self._multi_corruption.discrete_corruptions.values()} == {N}  # type: ignore

        self._predictors = {
            k: v(corruption=self._multi_corruption.corruptions[k], score_fn=None)
            for k, v in predictor_partials.items()
        }

        self._correctors = {
            k: v(
                corruption=self._multi_corruption.corruptions[k],
                n_steps=n_steps_corrector,
                score_fn=None,
            )
            for k, v in corrector_partials.items()
        }
        self._eps_t = eps_t
        self._n_steps_corrector = n_steps_corrector
        self._device = device

    @property
    def diffusion_module(self) -> DiffusionModule:
        return self._diffusion_module

    @property
    def _multi_corruption(self) -> MultiCorruption:
        return self._diffusion_module.corruption

    def _score_fn(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        return self._diffusion_module.score_fn(x, t)

    @classmethod
    def from_pl_module(cls, pl_module: DiffusionLightningModule, **kwargs) -> PredictorCorrector:
        return cls(diffusion_module=pl_module.diffusion_module, device=pl_module.device, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        fix_starter: bool = False,
        guidance_conf: GuidanceConf | None = None,
    ) -> SampleAndMean:
        return self._sample_maybe_record(
            conditioning_data,
            mask=mask,
            record=False,
            fix_starter=fix_starter,
            guidance_conf=guidance_conf,
        )[:2]

    @torch.no_grad()
    def sample_with_record(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        fix_starter: bool = False,
        guidance_conf: GuidanceConf | None = None,
    ) -> SampleAndMeanAndRecords:
        return self._sample_maybe_record(
            conditioning_data,
            mask=mask,
            record=True,
            fix_starter=fix_starter,
            guidance_conf=guidance_conf,
        )

    @torch.no_grad()
    def _sample_maybe_record(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        record: bool = False,
        fix_starter: bool = False,
        guidance_conf: GuidanceConf | None = None,
    ) -> SampleAndMeanAndMaybeRecords:
        if isinstance(self._diffusion_module, torch.nn.Module):
            self._diffusion_module.eval()
        mask = mask or {}
        conditioning_data = conditioning_data.to(self._device)
        mask = {k: v.to(self._device) for k, v in mask.items()}
        batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
        if fix_starter:
            batch = batch.replace(pos=conditioning_data.pos, cell=conditioning_data.cell)
        return self._denoise(
            batch=batch,
            mask=mask,
            record=record,
            guidance_conf=guidance_conf,
        )

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
        guidance_conf: GuidanceConf | None = None,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        cell_r_mean = self._predictors["cell"].corruption.get_limit_mean(x=batch.cell, batch=batch)
        batch = batch.replace(cell_r=batch.cell.clone(), cell_r_mean=cell_r_mean)

        start_step = self.N - guidance_conf.guidance_steps if guidance_conf is not None else self.N
        guidance_update = guidance_factory(
            guidance_method=guidance_conf.guidance_method,
            guidance_conf=guidance_conf,
            multi_corruption=self._multi_corruption,
            predictors=self._predictors,
        )

        for i in tqdm(range(self.N), miniters=50, mininterval=5, leave=False):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            # Corrector updates.
            if self._correctors and i <= start_step:
                for _ in range(self._n_steps_corrector):
                    score = self._score_fn(batch, t)
                    fns = {
                        k: corrector.step_given_score for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[str, tuple[torch.Tensor, torch.Tensor]] = apply(
                        fns=fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )

            # Predictor updates
            score = self._score_fn(batch, t)

            predictor_fns = {
                k: predictor.update_given_score for k, predictor in self._predictors.items()
            }

            samples_means = apply(
                fns=predictor_fns,
                x=batch,
                score=score,
                broadcast={"t": t, "batch": batch, "dt": dt},
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )

            if i > start_step:
                samples_means = guidance_update(
                    samples_means=samples_means,
                    batch=batch,
                    score=score,
                    t_step=i,
                    t_continuous=t,
                    dt=dt,
                    guidance_conf=guidance_conf,
                )

            if record:
                recorded_samples.append(batch.clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            )

        return batch, mean_batch, recorded_samples


def _mask_replace(
    samples_means: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch: BatchedData,
    mean_batch: BatchedData,
    mask: dict[str, torch.Tensor | None],
) -> SampleAndMean:
    # Apply masks
    samples_means = apply(
        fns=dict.fromkeys(samples_means, _mask_both),
        broadcast={},
        sample_and_mean=samples_means,
        mask=mask,
        old_x=batch,
    )

    # Put the updated values in `batch` and `mean_batch`
    batch = batch.replace(**{k: v[0] for k, v in samples_means.items()})
    mean_batch = mean_batch.replace(**{k: v[1] for k, v in samples_means.items()})
    return batch, mean_batch


def _mask_both(
    *, sample_and_mean: tuple[torch.Tensor, torch.Tensor], old_x: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(_mask(old_x=old_x, new_x=x, mask=mask) for x in sample_and_mean)  # type: ignore


def _mask(*, old_x: torch.Tensor, new_x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Replace new_x with old_x where mask is 1."""
    if mask is None:
        return new_x
    else:
        return new_x.lerp(old_x, mask)


def _sample_prior(
    multi_corruption: MultiCorruption,
    conditioning_data: BatchedData,
    mask: Mapping[str, torch.Tensor] | None,
) -> BatchedData:
    samples = {
        k: multi_corruption.corruptions[k]
        .prior_sampling(
            shape=conditioning_data[k].shape,
            conditioning_data=conditioning_data,
            batch_idx=conditioning_data.get_batch_idx(field_name=k),
        )
        .to(conditioning_data[k].device)
        for k in multi_corruption.corruptions
    }
    assert hasattr(multi_corruption.corruptions["cell"], "cell_mean_sampling"), (
        "Guided sampling requires `cell_mean_sampling` in the corruption"
    )
    samples["cell_mean"] = multi_corruption.corruptions["cell"].cell_mean_sampling(
        shape=conditioning_data["cell"].shape,
        conditioning_data=conditioning_data,
        batch_idx=conditioning_data.get_batch_idx(field_name="cell"),
    )
    mask = mask or {}
    for k, msk in mask.items():
        if k in multi_corruption.corrupted_fields:
            samples[k].lerp_(conditioning_data[k], msk)
    return conditioning_data.replace(**samples)
