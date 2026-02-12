from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # TODO: Clip gradients if max_grad_norm is set
            if group["max_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    group["params"], group["max_grad_norm"]
                )  # normlaize gradients into smaller sizegroup["params"], group["max_grad_norm"]

            for param in group["params"]:  # loop each parameter
                if param.grad is None:
                    continue
                gradient = param.grad.data  # focus gradient
                if gradient.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[param]  # each weight has its own memory

                # Initialize state if it does not exist
                if len(state) == 0:
                    state["step"] = 0
                    state["average_direction"] = torch.zeros_like(param.data)
                    state["average_size"] = torch.zeros_like(param.data)

                average_direction = state[
                    "average_direction"
                ]  # exponential moving average of past gradients, find trend
                average_size = state[
                    "average_size"
                ]  # exponential moving average of squared gradients, big gradient weights explode, small gradient weights never learn

                # TODO: Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group[
                    "betas"
                ]  # how much much do we trust the past direction, used for momentum; how stable is the gradient magnitude, prevents overreacting to peaks
                eps = group["eps"]  # division by zero
                weight_decay = group["weight_decay"]  # shrink weights over time
                correct_bias = group["correct_bias"]  # improves early learning speed

                state["step"] += 1  # step counter

                # TODO: Update first and second moments of the gradients
                average_direction.mul_(beta1).add_(
                    gradient, alpha=1 - beta1
                )  # momentum
                average_size.mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )  # variance, how big gradients usually are, how risky it is

                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in Algorithm 2
                # https://arxiv.org/pdf/1711.05101
                if correct_bias:  # adam move too slowly at the beginning
                    momentum_compensated = 1 - beta1 ** state["step"]
                    variance_compensated = 1 - beta2 ** state["step"]
                    step_size = (
                        alpha * (variance_compensated**0.5) / momentum_compensated
                    )
                else:
                    step_size = alpha

                # TODO: Update parameters
                denominator = average_size.sqrt().add_(eps)  # gradient scale, not 0
                param.data.addcdiv_(
                    average_direction, denominator, value=-step_size
                )  # update params

                # TODO: Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if weight_decay > 0.0:
                    param.data.add_(
                        param.data, alpha=-alpha * weight_decay
                    )  # decay weights
        return loss
