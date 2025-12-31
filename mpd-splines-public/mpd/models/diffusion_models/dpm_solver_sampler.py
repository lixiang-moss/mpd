import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from mpd.models.diffusion_models.sample_functions import apply_hard_conditioning, extract, guide_gradient_steps
from torch_robotics.torch_utils.torch_timer import TimerCUDA

# -----------------------------------------------------------------------------#
#                                DPM-Solver utils                              #
# -----------------------------------------------------------------------------#


def _import_dpm_solver():
    """
    Import the DPM-Solver implementation that is provided in the workspace root
    under `dpm_solver/dpm_solver_pytorch.py`.
    """
    try:
        from dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP  # type: ignore
    except ModuleNotFoundError:
        workspace_root = Path(__file__).resolve().parents[4]
        solver_path = workspace_root / "dpm_solver"
        sys.path.append(str(solver_path))
        from dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP  # type: ignore
    return DPM_Solver, NoiseScheduleVP


_DPM_SOLVER_CACHE = None


@torch.no_grad()
def dpm_solver_sample_loop(
    diffusion_model,
    shape_x: Tuple[int, ...],
    hard_conds,
    context_d: Optional[dict] = None,
    return_chain: bool = False,
    return_chain_x_recon: bool = False,
    # DPM-Solver options
    dpm_solver_steps: int = 10,
    dpm_solver_order: int = 2,
    dpm_solver_algorithm_type: str = "dpmsolver++",
    dpm_solver_method: str = "multistep",
    dpm_solver_skip_type: str = "time_uniform",
    dpm_solver_solver_type: str = "dpmsolver",
    dpm_solver_denoise_to_zero: bool = False,
    lower_order_final: bool = True,
    use_continuous_time: bool = False,
    # Guidance options
    guide: Optional[Callable] = None,
    guide_lr: float = 0.05,
    n_guide_steps: int = 1,
    max_perturb_x: float = 0.1,
    clip_grad: bool = False,
    clip_grad_rule: str = "value",
    max_grad_norm: float = 1.0,
    max_grad_value: float = 1.0,
    scale_grad_by_std: bool = False,
    t_start_guide: int = 0,
    prior_weight_with_guide: float = 1.0,
    compute_costs_with_xrecon: bool = False,
    results_ns=None,
    debug: bool = False,
    **kwargs,
):
    """
    Sampling loop that wraps the official DPM-Solver implementation while keeping
    the MPD guidance/conditioning hooks used by DDPM/DDIM samplers.
    """
    device = diffusion_model.betas.device
    batch_size = shape_x[0]

    # Lazy import to avoid impacting DDPM/DDIM if DPM-Solver is unavailable.
    global _DPM_SOLVER_CACHE
    if _DPM_SOLVER_CACHE is None:
        _DPM_SOLVER_CACHE = _import_dpm_solver()
    DPM_Solver, NoiseScheduleVP = _DPM_SOLVER_CACHE

    # context encoder
    context_emb = None
    if context_d is not None and diffusion_model.context_model is not None:
        context_emb = diffusion_model.context_model(**context_d)

    # noise schedule built from the trained diffusion model (discrete schedule)
    noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=diffusion_model.alphas_cumprod)
    total_n = diffusion_model.n_diffusion_steps

    def _to_batch_time(t_continuous: torch.Tensor) -> torch.Tensor:
        """
        DPM-Solver passes scalar `t` (0-d tensor) at each step, but MPD components
        expect a vector of size batch_size.
        """
        if not torch.is_tensor(t_continuous):
            t_continuous = torch.tensor(t_continuous, device=device, dtype=torch.float32)
        if t_continuous.ndim == 0:
            return t_continuous.expand(batch_size)
        if t_continuous.ndim == 1 and t_continuous.shape[0] == 1 and batch_size != 1:
            return t_continuous.expand(batch_size)
        if t_continuous.ndim != 1 or t_continuous.shape[0] != batch_size:
            raise ValueError(
                f"Expected t to be scalar or shape ({batch_size},), got shape={tuple(t_continuous.shape)}"
            )
        return t_continuous

    def continuous_to_discrete_idx(t_continuous: torch.Tensor) -> torch.Tensor:
        """
        Map continuous time in [1/N, 1] to the discrete index the model expects.
        """
        t_continuous = _to_batch_time(t_continuous)
        return torch.clamp((t_continuous * total_n - 1).round(), 0, total_n - 1).long()

    def continuous_to_model_time(t_continuous: torch.Tensor) -> torch.Tensor:
        """
        Map continuous time in [1/N, 1] to the model's time scale [0, N-1] without rounding.
        """
        t_continuous = _to_batch_time(t_continuous)
        t_model = t_continuous * total_n - 1.0
        return torch.clamp(t_model, 0.0, float(total_n - 1))

    def predict_noise(x_in: torch.Tensor, t_continuous: torch.Tensor) -> torch.Tensor:
        """
        Predict noise epsilon for DPM-Solver, adapting the MPD diffusion model to
        continuous time.
        """
        t_discrete = continuous_to_discrete_idx(t_continuous)
        t_model = continuous_to_model_time(t_continuous) if use_continuous_time else t_discrete
        eps_pred = diffusion_model.model(x_in, t_model, context_emb)
        # Convert to noise if the model predicts x0 directly
        if not diffusion_model.predict_epsilon:
            eps_pred = diffusion_model.predict_noise_from_start(x_in, t_discrete, eps_pred)
        if guide is not None:
            eps_pred = prior_weight_with_guide * eps_pred
        return eps_pred

    # track guidance compute time separately
    t_guide_accum = 0.0

    def correcting_xt_fn(x_in: torch.Tensor, t_continuous: torch.Tensor, step: int) -> torch.Tensor:
        """
        Projection of hard conditions + optional cost-guide gradient steps.
        """
        nonlocal t_guide_accum
        x_proj = apply_hard_conditioning(x_in, hard_conds)

        apply_guidance = guide is not None and (dpm_solver_steps - step) <= t_start_guide
        if apply_guidance:
            with TimerCUDA() as t_guide:
                t_discrete = continuous_to_discrete_idx(t_continuous)
                model_var = extract(diffusion_model.posterior_variance, t_discrete, x_proj.shape)
                x_proj = guide_gradient_steps(
                    x_proj,
                    t=t_discrete,
                    model=diffusion_model,
                    hard_conds=hard_conds,
                    context_d=context_d,
                    guide=guide,
                    guide_lr=guide_lr,
                    n_guide_steps=n_guide_steps,
                    max_perturb_x=max_perturb_x,
                    clip_grad=clip_grad,
                    clip_grad_rule=clip_grad_rule,
                    max_grad_norm=max_grad_norm,
                    max_grad_value=max_grad_value,
                    scale_grad_by_std=scale_grad_by_std,
                    model_var=model_var,
                    compute_costs_with_xrecon=compute_costs_with_xrecon,
                    debug=debug,
                )
            t_guide_accum += t_guide.elapsed

        return apply_hard_conditioning(x_proj, hard_conds)

    x = torch.randn(shape_x, device=device)
    x = apply_hard_conditioning(x, hard_conds)

    dpm_solver = DPM_Solver(
        predict_noise,
        noise_schedule,
        algorithm_type=dpm_solver_algorithm_type,
        correcting_xt_fn=correcting_xt_fn,
    )

    return_intermediate = return_chain or return_chain_x_recon

    with TimerCUDA() as t_generator:
        if return_intermediate:
            x, intermediates = dpm_solver.sample(
                x,
                steps=dpm_solver_steps,
                t_start=noise_schedule.T,
                t_end=1.0 / noise_schedule.total_N,
                order=dpm_solver_order,
                skip_type=dpm_solver_skip_type,
                method=dpm_solver_method,
                lower_order_final=lower_order_final,
                denoise_to_zero=dpm_solver_denoise_to_zero,
                solver_type=dpm_solver_solver_type,
                return_intermediate=True,
            )
        else:
            x = dpm_solver.sample(
                x,
                steps=dpm_solver_steps,
                t_start=noise_schedule.T,
                t_end=1.0 / noise_schedule.total_N,
                order=dpm_solver_order,
                skip_type=dpm_solver_skip_type,
                method=dpm_solver_method,
                lower_order_final=lower_order_final,
                denoise_to_zero=dpm_solver_denoise_to_zero,
                solver_type=dpm_solver_solver_type,
                return_intermediate=False,
            )
            intermediates = [x]

    if results_ns is not None:
        results_ns.t_generator = getattr(results_ns, "t_generator", 0.0) + t_generator.elapsed
        results_ns.t_guide = getattr(results_ns, "t_guide", 0.0) + t_guide_accum

    chain: List[torch.Tensor] = intermediates if return_intermediate else [x]
    if return_chain:
        chain = [apply_hard_conditioning(x_step, hard_conds) for x_step in chain]
        chain_tensor = torch.stack(chain, dim=1)
    else:
        chain_tensor = None

    if return_chain_x_recon:
        # We do not have intermediate x_recon values from DPM-Solver; reuse x_t states
        chain_x_recon_tensor = chain_tensor.clone() if chain_tensor is not None else x.unsqueeze(1)
    else:
        chain_x_recon_tensor = None

    outputs = [x]
    if return_chain:
        outputs.append(chain_tensor)
    if return_chain_x_recon:
        outputs.append(chain_x_recon_tensor)
    return tuple(outputs)
