"""ROLE: EDA ENGINEER | COMMON EDA TOOLKIT - ATTRIBUTION
Integrated gradients for the Stage 2 forecasters: how much of a forecast comes from each input channel,
each past hour and each node. Works on any PyTorch model that takes several input tensors.

OWNER : Role 2 - EDA Engineer
READS : nothing (the caller passes the model and its input tensors)
WRITES: nothing

Integrated gradients in one sentence: walk in small steps from a "nothing happening" baseline to the real
input, add up the gradients along the way, and multiply by (input - baseline). The attributions of all inputs
add up to f(input) - f(baseline) (the "completeness" check), which tells us the numbers are not made up.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch


def integrated_gradients(fn: Callable[..., torch.Tensor], inputs: Dict[str, torch.Tensor],
                         baselines: Dict[str, torch.Tensor] | None = None, steps: int = 32) -> Dict[str, torch.Tensor]:
    """Integrated-gradients attribution of the scalar-per-sample output fn(**inputs) -> [B] to every named input.

    Inputs listed in `inputs` are attributed; pass anything that must stay fixed (e.g. the anchor) inside `fn`.
    Baselines default to all zeros, which for robust-scaled channels means "every channel at its node's median".
    """
    if baselines is None:
        baselines = {k: torch.zeros_like(v) for k, v in inputs.items()}
    totals = {k: torch.zeros_like(v) for k, v in inputs.items()}
    # Midpoint rule: evaluate the gradient at the centre of each of the `steps` slices between baseline and input.
    for s in range(steps):
        alpha = (s + 0.5) / steps
        point = {k: (baselines[k] + alpha * (inputs[k] - baselines[k])).detach().requires_grad_(True) for k in inputs}
        out = fn(**point).sum()
        grads = torch.autograd.grad(out, list(point.values()), allow_unused=True)
        for k, g in zip(point.keys(), grads):
            if g is not None:
                totals[k] += g.detach()
    # Average gradient along the path x (input - baseline) = the attribution of each input value.
    return {k: (inputs[k] - baselines[k]) * totals[k] / steps for k in inputs}


def completeness_gap(fn: Callable[..., torch.Tensor], inputs: Dict[str, torch.Tensor],
                     attributions: Dict[str, torch.Tensor], baselines: Dict[str, torch.Tensor] | None = None) -> float:
    """Relative gap between sum(attributions) and f(input) - f(baseline); close to 0 means the attribution is trustworthy."""
    if baselines is None:
        baselines = {k: torch.zeros_like(v) for k, v in inputs.items()}
    with torch.no_grad():
        delta = (fn(**inputs) - fn(**baselines)).sum().item()
    total = sum(a.sum().item() for a in attributions.values())
    return float(abs(total - delta) / (abs(delta) + 1e-8))
