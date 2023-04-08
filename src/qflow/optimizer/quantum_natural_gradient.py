from typing import Callable

import numpy as np
import pennylane as qml
from scipy.linalg import lstsq

from qflow.abstract_optimizer import AbstractOptimizer


class QNG2Optimizer(AbstractOptimizer):
    """
    An implementation of the quantum natural gradient optimizer (QNG).
        https://arxiv.org/abs/1909.02108

    The QNG2 optimizer is a variation of the QNG optimization algorithm that solves
    the system of equations using least squares instead of a pseudo inverse.
    Moreover, the pennylane QNG implementation cannot handle singular matrices.
    This implementation adds an offdiagonal term to the metric tensor to avoid singular matrices.

    For pure states quantum imaginary time evolution (QITE) is equivalent with
    quantum natural gradient (QNG) optimization up to a constant rescaling.

    $$
        \text { QFIM }_{i, j} = 4 * \text { metric_tensor }_{i, j}=4*\operatorname{Re}\left[\left\langle\partial_i \psi(\theta) \mid \partial_j \psi(\theta)\right\rangle-\left\langle\partial_i \psi(\theta) \mid \psi(\theta)\right\rangle\left\langle\psi(\theta) \mid \partial_j \psi(\theta)\right\rangle\right]
    $$

    Args:
        stepsize (float, optional): The learning rate for gradient descent. Defaults to 0.01.
        approx (str, optional): The approximation method for the metric tensor.
            Defaults to "block-diag".

    Attributes:
        stepsize (float): The learning rate for gradient descent.
        grad_fn (Callable): The gradient function for the objective function.
        approx (str): The approximation method for the metric tensor.
        F (np.ndarray): The metric tensor.
        grad (np.ndarray): The gradient of the objective function.
        nat_grad (np.ndarray): The natural gradient of the objective function.
    """

    def __init__(self, stepsize: float = 0.01, approx: str = "block-diag"):
        self.stepsize = stepsize
        self.grad_fn = None
        self.approx = approx
        self.F = None

    def step(
        self,
        objective_fn: Callable,
        params: np.ndarray,
        grad_fn: Callable = None,
        *args,
        **kwargs
    ):
        if self.grad_fn == None:
            self.grad_fn = qml.grad(objective_fn)

        params_shape = params.shape
        params = params.reshape((-1,))

        if self.F is None:
            self.F = qml.metric_tensor(objective_fn, approx=self.approx)(params)
        # https://discuss.pennylane.ai/t/quantum-natural-gradient-descent/351/2
        # Instead of pseudo inverse solve system of equations
        # In the current implmentation of the QNG optimizer, adding small values
        # to the diagonal is neglected.
        # self.F += np.identity(self.F.shape[0]) * 0.01
        # pennylane's qng uses np.linalg.solve to solve the system of equations
        # nat_grad = np.linalg.solve(self.F, self.grad_fn(params))
        self.grad = self.grad_fn(params)
        self.nat_grad, _, _, _ = lstsq(self.F, self.grad, cond=1e-7)
        params -= self.stepsize * self.nat_grad

        params = params.reshape(params_shape)
        return params

    def step_and_cost(
        self,
        objective_fn: Callable,
        params: np.ndarray,
        grad_fn: Callable = None,
        *args,
        **kwargs
    ):
        params = self.step(objective_fn, params, grad_fn, *args, **kwargs)
        cost = objective_fn(params)

        return params, cost
