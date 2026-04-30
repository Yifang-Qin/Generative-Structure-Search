from dataclasses import dataclass
from typing import Literal

from mattersim.forcefield.potential import Potential

DEFAULT_GUIDANCE_PARAMS = {
    None: {"pos_guide_coef": 5e-5, "cell_guide_coef": 2e-1, "guidance_steps": 0},
    "diffusion": {"pos_guide_coef": 0, "cell_guide_coef": 0, "guidance_steps": 0},
    "Mi": {"pos_guide_coef": 5e-5, "cell_guide_coef": 2e-1, "guidance_steps": 750},
    "MN": {"pos_guide_coef": 1e-4, "cell_guide_coef": 1e-1, "guidance_steps": 400},
    "M_tau": {"pos_guide_coef": 1e-4, "cell_guide_coef": 1e-1, "guidance_steps": 400},
}


@dataclass
class GuidanceConf:
    potential: Potential

    guidance_method: Literal["Mi", "MN", "M_tau", "diffusion"] | None = None
    pos_guide_coef: float | None = None
    cell_guide_coef: float | None = None
    guidance_steps: int | None = None

    # sigmoid-like Mi guidance
    t_mid: int = 750
    t_scale: int = 250

    # MN and M_tau guidance
    n_iter: int = 1
    n_recur: int = 1
    tfg_rho: float = 0.0
    tfg_gamma: float = 0.0
    tfg_mu: float = 2.5

    tau_guidance: bool = False
    tau: float = 0.5

    def __post_init__(self) -> None:
        defaults = DEFAULT_GUIDANCE_PARAMS.get(self.guidance_method, DEFAULT_GUIDANCE_PARAMS[None])

        if self.pos_guide_coef is None:
            self.pos_guide_coef = defaults["pos_guide_coef"]

        if self.cell_guide_coef is None:
            self.cell_guide_coef = defaults["cell_guide_coef"]

        if self.guidance_steps is None:
            self.guidance_steps = defaults["guidance_steps"]
