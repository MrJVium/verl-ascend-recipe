"""DR.Kernel's MRS per-token catastrophic-ratio veto.

The veto is a tiny extension of verl's ``rollout_correction`` subsystem: any
sequence containing a token whose ``pi_train / pi_rollout`` drops below
``rollout_token_veto_threshold`` (paper value: 1e-4) has its entire
``response_mask`` row zeroed for the gradient step. It composes with
``rollout_rs`` purely by mask multiplication, so applying it as a post-step of
the upstream rejection mask is mathematically identical to baking it into the
helper.
"""

import math
from typing import Optional

import torch

import verl.trainer.ppo.core_algos as core_algos
import verl.trainer.ppo.rollout_corr_helper as rollout_corr_helper
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_policy_loss

# Capture the upstream implementations *before* we override the registry, so the
# wrapper can delegate to the real bypass loss and restore the real helper.
_UPSTREAM_BYPASS_LOSS = core_algos.compute_policy_loss_bypass_mode
_UPSTREAM_CORR_AND_RS = rollout_corr_helper.compute_rollout_correction_and_rejection_mask

# Metric keys carry the same "rollout_corr/" prefix the upstream helper applies
# to every metric in its final pass, so the veto metrics sit alongside the rest.
_VETO_SEQ_FRACTION_KEY = "rollout_corr/rollout_veto_seq_fraction"
_VETO_TOKEN_FRACTION_KEY = "rollout_corr/rollout_veto_catastrophic_token_fraction"


def _apply_token_veto(
    modified_response_mask: torch.Tensor,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_token_veto_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Zero the full row of any sequence with a catastrophic-ratio token.

    """
    log_ratio = old_log_prob - rollout_log_prob
    log_veto_threshold = math.log(rollout_token_veto_threshold)
    catastrophic_tokens = (log_ratio < log_veto_threshold) & response_mask.bool()
    has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)
    veto_mask = (~has_catastrophic).to(dtype=modified_response_mask.dtype)
    modified_response_mask = modified_response_mask * veto_mask
    metrics = {
        _VETO_SEQ_FRACTION_KEY: has_catastrophic.float().mean().item(),
        _VETO_TOKEN_FRACTION_KEY: verl_F.masked_mean(catastrophic_tokens.float(), response_mask).item(),
    }
    return modified_response_mask, metrics


def _read_veto_threshold(config) -> Optional[float]:
    """Pull ``rollout_token_veto_threshold`` out of the rollout_correction config.

    """
    if config is None or not hasattr(config, "policy_loss"):
        return None
    rollout_corr_config = config.policy_loss.get("rollout_correction", None)
    if rollout_corr_config is None:
        return None
    return rollout_corr_config.get("rollout_token_veto_threshold", None)


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode_with_veto(*args, config=None, **kwargs):
    """``bypass_mode`` policy loss + DR.Kernel per-token catastrophic-ratio veto.

    Overrides the upstream ``bypass_mode`` registration. When no veto threshold
    is configured this is a transparent passthrough to the upstream loss.
    """
    veto_threshold = _read_veto_threshold(config)
    if not veto_threshold:
        return _UPSTREAM_BYPASS_LOSS(*args, config=config, **kwargs)
    if veto_threshold <= 0:
        raise ValueError(f"rollout_token_veto_threshold must be positive, got {veto_threshold}.")

    def _helper_with_veto(*h_args, **h_kwargs):
        proto, mask, metrics = _UPSTREAM_CORR_AND_RS(*h_args, **h_kwargs)
        mask, veto_metrics = _apply_token_veto(
            modified_response_mask=mask,
            old_log_prob=h_kwargs["old_log_prob"],
            rollout_log_prob=h_kwargs["rollout_log_prob"],
            response_mask=h_kwargs["response_mask"],
            rollout_token_veto_threshold=veto_threshold,
        )
        metrics.update(veto_metrics)
        return proto, mask, metrics

    # The upstream bypass loss imports the helper by name from the module at call
    # time, so patching the module attribute before delegating is enough. The
    # actor computes the loss single-threaded per micro-batch, so the global
    # swap is safe; the finally clause restores it unconditionally.
    rollout_corr_helper.compute_rollout_correction_and_rejection_mask = _helper_with_veto
    try:
        return _UPSTREAM_BYPASS_LOSS(*args, config=config, **kwargs)
    finally:
        rollout_corr_helper.compute_rollout_correction_and_rejection_mask = _UPSTREAM_CORR_AND_RS
