"""Main module defining RGD optimizer."""
from math import sqrt

import torch
import torch.optim as optim


class AutonomousOptimizer(optim.Optimizer):
    def __init__(self, params, history=25):
        defaults = dict(history=history)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            history = group["history"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g_k = p.grad

                # Get v_k if present, if not, we are on first iteration and set to zero
                param_state = self.state[p]
                v_k = param_state.get("v_k", torch.zeros_like(g_k))

                policy()
                if integrator == "symplectic_euler":
                    # v_{k+1} = momentum * v_k - lr * g_k
                    v_k.mul_(momentum).add_(g_k, alpha=-lr)

                    # x_{k+1} = x_k + v_{k+1} / sqrt(delta * ||v_k||^2 + 1)
                    norm_factor = sqrt(delta * torch.square(v_k).sum() + 1)
                    p.add_(v_k, alpha=1 / norm_factor)

                else:  # integrator == "leapfrog"
                    # v_{k+1/2} = sqrt(momentum) * v_k - lr * g_k
                    v_k.mul_(sqrt(momentum)).add_(g_k, alpha=-lr)

                    # x_{k+1} = x_{k+1/2} + v_{k+1/2} / sqrt(delta *||v_{k+1/2}||^2 + 1)
                    norm_factor = sqrt(delta * torch.square(v_k).sum() + 1)
                    p.add_(v_k, alpha=1 / norm_factor)

                    # v_{k+1} = sqrt(momentum) * v_{k+1/2}
                    v_k.mul_(sqrt(momentum))

                    # x_{k+3/2} = x_{k+1} + sqrt(momentum) *
                    #       v_{k+1} / sqrt(momentum * delta * ||v_{k+1}||^2 + 1)
                    norm_factor = sqrt(momentum * delta * torch.square(v_k).sum() + 1)
                    p.add_(v_k, alpha=sqrt(momentum) / norm_factor)

                param_state["v_k"] = v_k

        return loss
