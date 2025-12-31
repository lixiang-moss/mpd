import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from mpd.models.diffusion_models.sample_functions import apply_hard_conditioning, extract, guide_gradient_steps
from torch_robotics.torch_utils.torch_timer import TimerCUDA

# -----------------------------------------------------------------------------#
#                                  UniPC utils                                 #
# -----------------------------------------------------------------------------#


def _import_unipc_noise_schedule():
    """
    Import UniPC's NoiseScheduleVP implementation from the workspace root.
    """
    try:
        from uni_pc import NoiseScheduleVP  # type: ignore
    except ModuleNotFoundError:
        workspace_root = Path(__file__).resolve().parents[4]
        unipc_path = workspace_root / "unipc"
        sys.path.append(str(unipc_path))
        from uni_pc import NoiseScheduleVP  # type: ignore
    return NoiseScheduleVP


_UNIPC_NOISE_SCHEDULE_CACHE = None


class ModelOutputBuffer:
    """History buffer for multistep UniPC updates."""

    def __init__(self, max_order: int):
        self.max_order = max_order
        self._model_outputs: List[torch.Tensor] = []
        self._times: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self._model_outputs)

    def add(self, t: torch.Tensor, model_out: torch.Tensor) -> None:
        self._times.append(t)
        self._model_outputs.append(model_out)
        if len(self._model_outputs) > self.max_order:
            self._model_outputs.pop(0)
            self._times.pop(0)

    def get_recent(self, order: int, step_idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Cold start: never read from buffer[-1] when step_idx == 0.
        assert step_idx > 0, "UniPC warm-up requires step_idx > 0 before accessing history."
        assert order <= len(self._model_outputs), "Requested order exceeds available history."
        return self._model_outputs[-order:], self._times[-order:]


def _to_device_dtype(value, x: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=x.device, dtype=x.dtype)
    return torch.tensor(value, device=x.device, dtype=x.dtype)


def _to_scalar(value, x: torch.Tensor, name: str) -> torch.Tensor:
    tensor = _to_device_dtype(value, x)
    if tensor.numel() != 1:
        raise ValueError(f"Expected scalar {name}, got shape={tuple(tensor.shape)}")
    return tensor.reshape(())


def _maybe_clip_x0(x0: torch.Tensor, clip_denoised: bool, clip_min: float, clip_max: float) -> torch.Tensor:
    if not clip_denoised:
        return x0
    return torch.clamp(x0, min=clip_min, max=clip_max)


def _broadcast_time_coeff(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Explicitly reshape time coefficients for safe broadcasting over x.
    """
    coeff = _to_device_dtype(coeff, x)
    if coeff.ndim == 0:
        coeff = coeff.view(1)
    if coeff.ndim != 1:
        raise ValueError(f"Expected time coeff to be scalar or (B,), got shape={tuple(coeff.shape)}")
    return coeff.view(-1, *([1] * (x.dim() - 1)))


def _combine_d1s(d1s: torch.Tensor, weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Weighted sum over the history axis (K) for residuals.
    """
    weights = _to_device_dtype(weights, x)
    view_shape = (1, -1) + (1,) * (x.dim() - 1)
    return torch.sum(d1s * weights.view(view_shape), dim=1)


def _get_time_steps(noise_schedule, skip_type: str, t_T: float, t_0: float, steps: int, device: torch.device):
    if skip_type == "logSNR":
        lambda_T = noise_schedule.marginal_lambda(torch.tensor(t_T, device=device))
        lambda_0 = noise_schedule.marginal_lambda(torch.tensor(t_0, device=device))
        logsnr_steps = torch.linspace(lambda_T.item(), lambda_0.item(), steps + 1, device=device)
        return noise_schedule.inverse_lambda(logsnr_steps)
    if skip_type == "time_uniform":
        return torch.linspace(t_T, t_0, steps + 1, device=device)
    if skip_type == "time_quadratic":
        t_order = 2
        return torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), steps + 1, device=device).pow(
            t_order
        )
    raise ValueError(f"Unsupported skip_type {skip_type}")


def _multistep_uni_pc_bh_update(
    x: torch.Tensor,
    model_prev_list: List[torch.Tensor],
    t_prev_list: List[torch.Tensor],
    t: torch.Tensor,
    order: int,
    noise_schedule,
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    predict_x0: bool,
    variant: str,
    use_corrector: bool,
    x_t: Optional[torch.Tensor] = None,
):
    assert order <= len(model_prev_list)

    t_prev_0 = t_prev_list[-1]
    lambda_prev_0 = _to_scalar(noise_schedule.marginal_lambda(t_prev_0), x, "lambda_prev_0")
    lambda_t = _to_scalar(noise_schedule.marginal_lambda(t), x, "lambda_t")
    h = lambda_t - lambda_prev_0
    assert torch.all(h != 0), "Zero log-SNR step detected; check time grid construction."

    model_prev_0 = model_prev_list[-1]
    sigma_prev_0 = _to_device_dtype(noise_schedule.marginal_std(t_prev_0), x)
    sigma_t = _to_device_dtype(noise_schedule.marginal_std(t), x)
    log_alpha_prev_0 = _to_device_dtype(noise_schedule.marginal_log_mean_coeff(t_prev_0), x)
    log_alpha_t = _to_device_dtype(noise_schedule.marginal_log_mean_coeff(t), x)
    alpha_t = torch.exp(log_alpha_t)

    rks = []
    d1s = []
    for i in range(1, order):
        t_prev_i = t_prev_list[-(i + 1)]
        model_prev_i = model_prev_list[-(i + 1)]
        lambda_prev_i = _to_scalar(noise_schedule.marginal_lambda(t_prev_i), x, "lambda_prev_i")
        rk = (lambda_prev_i - lambda_prev_0) / h
        rks.append(rk)
        d1s.append((model_prev_i - model_prev_0) / rk)

    rks.append(torch.ones_like(h))
    rks = torch.stack(rks)

    hh = -h if predict_x0 else h
    h_phi_1 = torch.expm1(hh)  # exp(h) - 1
    h_phi_k = h_phi_1 / hh - 1.0

    factorial_i = 1.0
    if variant == "bh1":
        B_h = hh
    elif variant == "bh2":
        B_h = torch.expm1(hh)
    else:
        raise NotImplementedError(f"Unsupported UniPC variant: {variant}")

    R = []
    b = []
    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i

    R = torch.stack(R)
    b = torch.stack(b)

    use_predictor = len(d1s) > 0 and x_t is None
    d1s_stack = None
    if len(d1s) > 0:
        d1s_stack = torch.stack(d1s, dim=1)
        if x_t is None:
            if order == 2:
                rhos_p = torch.tensor([0.5], device=x.device, dtype=x.dtype)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
    if use_corrector:
        if order == 1:
            rhos_c = torch.tensor([0.5], device=x.device, dtype=x.dtype)
        else:
            rhos_c = torch.linalg.solve(R, b)

    model_t = None
    if predict_x0:
        # Analytical linear term from the semi-linear ODE; do not use Euler.
        sigma_ratio = sigma_t / sigma_prev_0
        sigma_ratio_b = _broadcast_time_coeff(sigma_ratio, x)
        alpha_h_phi = alpha_t * h_phi_1
        alpha_h_phi_b = _broadcast_time_coeff(alpha_h_phi, x)
        x_t_ = sigma_ratio_b * x - alpha_h_phi_b * model_prev_0

        if x_t is None:
            if use_predictor:
                pred_res = _combine_d1s(d1s_stack, rhos_p, x)
            else:
                pred_res = torch.zeros_like(x)
            coeff = alpha_t * B_h
            x_t = x_t_ - _broadcast_time_coeff(coeff, x) * pred_res

        if use_corrector:
            model_t = model_fn(x_t, t)
            if d1s_stack is not None:
                corr_res = _combine_d1s(d1s_stack, rhos_c[:-1], x)
            else:
                corr_res = torch.zeros_like(x)
            d1_t = model_t - model_prev_0
            coeff = alpha_t * B_h
            x_t = x_t_ - _broadcast_time_coeff(coeff, x) * (corr_res + _broadcast_time_coeff(rhos_c[-1], x) * d1_t)
    else:
        # Analytical linear term from the semi-linear ODE; do not use Euler.
        linear_coeff = torch.exp(log_alpha_t - log_alpha_prev_0)
        linear_coeff_b = _broadcast_time_coeff(linear_coeff, x)
        sigma_h_phi = sigma_t * h_phi_1
        sigma_h_phi_b = _broadcast_time_coeff(sigma_h_phi, x)
        x_t_ = linear_coeff_b * x - sigma_h_phi_b * model_prev_0

        if x_t is None:
            if use_predictor:
                pred_res = _combine_d1s(d1s_stack, rhos_p, x)
            else:
                pred_res = torch.zeros_like(x)
            coeff = sigma_t * B_h
            x_t = x_t_ - _broadcast_time_coeff(coeff, x) * pred_res

        if use_corrector:
            model_t = model_fn(x_t, t)
            if d1s_stack is not None:
                corr_res = _combine_d1s(d1s_stack, rhos_c[:-1], x)
            else:
                corr_res = torch.zeros_like(x)
            d1_t = model_t - model_prev_0
            coeff = sigma_t * B_h
            x_t = x_t_ - _broadcast_time_coeff(coeff, x) * (corr_res + _broadcast_time_coeff(rhos_c[-1], x) * d1_t)

    return x_t, model_t


def _multistep_uni_pc_vary_update(
    x: torch.Tensor,
    model_prev_list: List[torch.Tensor],
    t_prev_list: List[torch.Tensor],
    t: torch.Tensor,
    order: int,
    noise_schedule,
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    predict_x0: bool,
    use_corrector: bool,
):
    assert order <= len(model_prev_list)

    t_prev_0 = t_prev_list[-1]
    lambda_prev_0 = _to_scalar(noise_schedule.marginal_lambda(t_prev_0), x, "lambda_prev_0")
    lambda_t = _to_scalar(noise_schedule.marginal_lambda(t), x, "lambda_t")
    h = lambda_t - lambda_prev_0
    assert torch.all(h != 0), "Zero log-SNR step detected; check time grid construction."

    model_prev_0 = model_prev_list[-1]
    sigma_prev_0 = _to_device_dtype(noise_schedule.marginal_std(t_prev_0), x)
    sigma_t = _to_device_dtype(noise_schedule.marginal_std(t), x)
    log_alpha_prev_0 = _to_device_dtype(noise_schedule.marginal_log_mean_coeff(t_prev_0), x)
    log_alpha_t = _to_device_dtype(noise_schedule.marginal_log_mean_coeff(t), x)
    alpha_t = torch.exp(log_alpha_t)

    rks = []
    d1s = []
    for i in range(1, order):
        t_prev_i = t_prev_list[-(i + 1)]
        model_prev_i = model_prev_list[-(i + 1)]
        lambda_prev_i = _to_scalar(noise_schedule.marginal_lambda(t_prev_i), x, "lambda_prev_i")
        rk = (lambda_prev_i - lambda_prev_0) / h
        rks.append(rk)
        d1s.append((model_prev_i - model_prev_0) / rk)

    rks.append(torch.ones_like(h))
    rks = torch.stack(rks)
    K = len(rks)

    # Build C matrix from rks (scalar log-SNR ratios).
    C_cols = []
    col = torch.ones_like(rks)
    for k in range(1, K + 1):
        C_cols.append(col)
        col = col * rks / (k + 1)
    C = torch.stack(C_cols, dim=1)

    d1s_stack = None
    if len(d1s) > 0:
        d1s_stack = torch.stack(d1s, dim=1)
        C_inv_p = torch.linalg.inv(C[:-1, :-1])
        A_p = C_inv_p
    if use_corrector:
        C_inv = torch.linalg.inv(C)
        A_c = C_inv

    hh = -h if predict_x0 else h
    h_phi_1 = torch.expm1(hh)
    h_phi_ks = []
    factorial_k = 1.0
    h_phi_k = h_phi_1
    for k in range(1, K + 2):
        h_phi_ks.append(h_phi_k)
        h_phi_k = h_phi_k / hh - 1.0 / factorial_k
        factorial_k *= k + 1

    model_t = None
    if predict_x0:
        # Analytical linear term from the semi-linear ODE; do not use Euler.
        sigma_ratio = sigma_t / sigma_prev_0
        x_t_ = _broadcast_time_coeff(sigma_ratio, x) * x - _broadcast_time_coeff(alpha_t * h_phi_1, x) * model_prev_0

        x_t = x_t_
        if d1s_stack is not None:
            for k in range(K - 1):
                res = _combine_d1s(d1s_stack, A_p[k], x)
                x_t = x_t - _broadcast_time_coeff(alpha_t * h_phi_ks[k + 1], x) * res

        if use_corrector:
            model_t = model_fn(x_t, t)
            d1_t = model_t - model_prev_0
            x_t = x_t_
            if d1s_stack is not None:
                for k in range(K - 1):
                    res = _combine_d1s(d1s_stack, A_c[k][:-1], x)
                    x_t = x_t - _broadcast_time_coeff(alpha_t * h_phi_ks[k + 1], x) * res
            k_last = K - 2 if K > 1 else 0
            x_t = x_t - _broadcast_time_coeff(alpha_t * h_phi_ks[K], x) * (
                _broadcast_time_coeff(A_c[k_last][-1], x) * d1_t
            )
    else:
        # Analytical linear term from the semi-linear ODE; do not use Euler.
        linear_coeff = torch.exp(log_alpha_t - log_alpha_prev_0)
        x_t_ = _broadcast_time_coeff(linear_coeff, x) * x - _broadcast_time_coeff(sigma_t * h_phi_1, x) * model_prev_0

        x_t = x_t_
        if d1s_stack is not None:
            for k in range(K - 1):
                res = _combine_d1s(d1s_stack, A_p[k], x)
                x_t = x_t - _broadcast_time_coeff(sigma_t * h_phi_ks[k + 1], x) * res

        if use_corrector:
            model_t = model_fn(x_t, t)
            d1_t = model_t - model_prev_0
            x_t = x_t_
            if d1s_stack is not None:
                for k in range(K - 1):
                    res = _combine_d1s(d1s_stack, A_c[k][:-1], x)
                    x_t = x_t - _broadcast_time_coeff(sigma_t * h_phi_ks[k + 1], x) * res
            k_last = K - 2 if K > 1 else 0
            x_t = x_t - _broadcast_time_coeff(sigma_t * h_phi_ks[K], x) * (
                _broadcast_time_coeff(A_c[k_last][-1], x) * d1_t
            )

    return x_t, model_t


@torch.no_grad()
def unipc_sample_loop(
    diffusion_model,
    shape_x: Tuple[int, ...],
    hard_conds,
    context_d: Optional[dict] = None,
    return_chain: bool = False,
    return_chain_x_recon: bool = False,
    # UniPC options
    unipc_steps: int = 20,
    unipc_order: int = 2,
    unipc_skip_type: str = "logSNR",
    unipc_variant: str = "bh1",
    unipc_lower_order_final: bool = True,
    unipc_denoise_to_zero: bool = False,
    use_continuous_time: bool = False,
    clip_denoised: bool = False,
    clip_denoised_min: float = -1.0,
    clip_denoised_max: float = 1.0,
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
    MPD-friendly UniPC sampler with explicit log-SNR stepping and history buffering.
    """
    device = diffusion_model.betas.device
    batch_size = shape_x[0]

    assert unipc_steps >= 1, "UniPC requires at least 1 step."
    assert unipc_order >= 1, "UniPC order must be >= 1."
    assert unipc_steps >= unipc_order, "UniPC steps must be >= order."
    assert unipc_skip_type == "logSNR", "UniPC requires logSNR-aligned steps; do not assume t is linear in log-SNR."

    # Lazy import to avoid impacting other samplers if UniPC files are missing.
    global _UNIPC_NOISE_SCHEDULE_CACHE
    if _UNIPC_NOISE_SCHEDULE_CACHE is None:
        _UNIPC_NOISE_SCHEDULE_CACHE = _import_unipc_noise_schedule()
    NoiseScheduleVP = _UNIPC_NOISE_SCHEDULE_CACHE

    # Context encoder
    context_emb = None
    if context_d is not None and diffusion_model.context_model is not None:
        context_emb = diffusion_model.context_model(**context_d)

    # Noise schedule built from the trained diffusion model (discrete schedule).
    noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=diffusion_model.alphas_cumprod)
    total_n = diffusion_model.n_diffusion_steps

    def _to_batch_time(t_continuous: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t_continuous):
            t_continuous = torch.tensor(t_continuous, device=device, dtype=torch.float32)
        if t_continuous.ndim == 0:
            t_continuous = t_continuous.view(1)
        if t_continuous.ndim == 1 and t_continuous.shape[0] == 1 and batch_size != 1:
            t_continuous = t_continuous.expand(batch_size)
        if t_continuous.ndim != 1 or t_continuous.shape[0] != batch_size:
            raise ValueError(
                f"Expected t to be scalar or shape ({batch_size},), got shape={tuple(t_continuous.shape)}"
            )
        return t_continuous

    def continuous_to_discrete_idx(t_continuous: torch.Tensor) -> torch.Tensor:
        # Map continuous time in [1/N, 1] to discrete indices; log-SNR uses alphas_cumprod, not linear t.
        t_continuous = _to_batch_time(t_continuous)
        return torch.clamp((t_continuous * total_n - 1).round(), 0, total_n - 1).long()

    def continuous_to_model_time(t_continuous: torch.Tensor) -> torch.Tensor:
        """
        Map continuous time in [1/N, 1] to the model's time scale [0, N-1] without rounding.
        This enables true continuous-time evaluation while staying consistent with the training scale.
        """
        t_continuous = _to_batch_time(t_continuous)
        t_model = t_continuous * total_n - 1.0
        return torch.clamp(t_model, 0.0, float(total_n - 1))

    predict_x0 = not diffusion_model.predict_epsilon

    def model_fn(x_in: torch.Tensor, t_continuous: torch.Tensor) -> torch.Tensor:
        if use_continuous_time:
            t_model = continuous_to_model_time(t_continuous)
        else:
            t_model = continuous_to_discrete_idx(t_continuous)
        model_out = diffusion_model.model(x_in, t_model, context_emb)
        if predict_x0:
            model_out = _maybe_clip_x0(model_out, clip_denoised, clip_denoised_min, clip_denoised_max)
        if guide is not None:
            model_out = prior_weight_with_guide * model_out
        return model_out

    def denoise_to_zero_fn(x_in: torch.Tensor, t_continuous: torch.Tensor) -> torch.Tensor:
        # Predict x0 at t_end using the analytical formula from alpha/sigma.
        model_out = model_fn(x_in, t_continuous)
        if predict_x0:
            return model_out
        alpha_t = _to_device_dtype(noise_schedule.marginal_alpha(t_continuous), x_in)
        sigma_t = _to_device_dtype(noise_schedule.marginal_std(t_continuous), x_in)
        alpha_t_b = _broadcast_time_coeff(alpha_t, x_in)
        sigma_t_b = _broadcast_time_coeff(sigma_t, x_in)
        return (x_in - sigma_t_b * model_out) / alpha_t_b

    # Guidance timing accumulation
    t_guide_accum = 0.0

    def correcting_xt_fn(x_in: torch.Tensor, t_continuous: torch.Tensor, step: int) -> torch.Tensor:
        nonlocal t_guide_accum
        x_proj = apply_hard_conditioning(x_in, hard_conds)
        apply_guidance = guide is not None and (unipc_steps - step) <= t_start_guide
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

    return_intermediate = return_chain or return_chain_x_recon
    chain: List[torch.Tensor] = []
    if return_intermediate:
        chain.append(x.clone())

    t_0 = 1.0 / noise_schedule.total_N
    t_T = noise_schedule.T
    assert t_0 > 0 and t_T > 0, "Time range must be positive for discrete UniPC."

    timesteps = _get_time_steps(noise_schedule, unipc_skip_type, t_T, t_0, unipc_steps, device)
    assert timesteps.shape[0] == unipc_steps + 1, "Unexpected time grid length."

    buffer = ModelOutputBuffer(max_order=unipc_order)

    # step = 0 (cold start)
    t = timesteps[0]
    model_0 = model_fn(x, t)
    buffer.add(t, model_0)

    with TimerCUDA() as t_generator:
        for step_idx in range(1, unipc_steps + 1):
            t = timesteps[step_idx]

            # Order promotion: step 1 -> 1st order, step 2 -> 2nd order, then max order.
            base_order = min(unipc_order, step_idx)
            if unipc_lower_order_final:
                step_order = min(base_order, unipc_steps + 1 - step_idx)
            else:
                step_order = base_order

            model_prev_list, t_prev_list = buffer.get_recent(step_order, step_idx)

            use_corrector = step_idx < unipc_steps
            if unipc_variant in ("bh1", "bh2"):
                x, model_x = _multistep_uni_pc_bh_update(
                    x,
                    model_prev_list,
                    t_prev_list,
                    t,
                    step_order,
                    noise_schedule,
                    model_fn,
                    predict_x0,
                    unipc_variant,
                    use_corrector=use_corrector,
                )
            elif unipc_variant == "vary_coeff":
                x, model_x = _multistep_uni_pc_vary_update(
                    x,
                    model_prev_list,
                    t_prev_list,
                    t,
                    step_order,
                    noise_schedule,
                    model_fn,
                    predict_x0,
                    use_corrector=use_corrector,
                )
            else:
                raise ValueError(f"Unknown UniPC variant: {unipc_variant}")

            x = correcting_xt_fn(x, t, step_idx)
            if return_intermediate:
                chain.append(x.clone())

            if step_idx < unipc_steps:
                if model_x is None:
                    model_x = model_fn(x, t)
                buffer.add(t, model_x)

        if unipc_denoise_to_zero:
            t = torch.tensor(t_0, device=device)
            x = denoise_to_zero_fn(x, t)
            x = correcting_xt_fn(x, t, unipc_steps + 1)
            if return_intermediate:
                chain.append(x.clone())

    if results_ns is not None:
        results_ns.t_generator = getattr(results_ns, "t_generator", 0.0) + t_generator.elapsed
        results_ns.t_guide = getattr(results_ns, "t_guide", 0.0) + t_guide_accum

    if return_intermediate:
        chain_tensor = torch.stack(chain, dim=1)
    else:
        chain_tensor = None

    if return_chain_x_recon:
        # UniPC does not return per-step x_recon; reuse x_t states like DPM-Solver.
        chain_x_recon_tensor = chain_tensor.clone() if chain_tensor is not None else x.unsqueeze(1)
    else:
        chain_x_recon_tensor = None

    outputs = [x]
    if return_chain:
        outputs.append(chain_tensor)
    if return_chain_x_recon:
        outputs.append(chain_x_recon_tensor)
    return tuple(outputs)
