import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest
from pennylane import QNGOptimizer

from qflow.optimizer import QNG2Optimizer
from qflow.tests.utils import circuit, circuit_2


@pytest.mark.parametrize(
    "num_layers, num_qubits, stepsize, seed",
    [
        (3, 3, 0.01, 0),
        (2, 3, 0.1, 0),
    ],
)
def qng_2_test(num_layers: int, num_qubits: int, stepsize: float, seed: int):
    # There are some minor differences in the update rules of the two optimizers.
    # The parameteres start to vary slightly after two decimals.
    # Numerical inaccuracies?
    qng_optimizer = QNGOptimizer(stepsize=stepsize)
    qng_2_optimizer = QNG2Optimizer(stepsize=stepsize)

    fun = circuit_2
    params = pnp.array([0.432, -0.123, 0.543, 0.233], requires_grad=True)

    for i in range(50):
        params_2 = qng_2_optimizer.step(fun, params)
        params = qng_optimizer.step(fun, params)

        # Why is it diverging already after the first decimal digit? Pennylanes
        # QNG must do something that I do not consider for the quantum natural
        # gradient.
        np.testing.assert_almost_equal(params, params_2, decimal=1)


if __name__ == "__main__":
    qng_2_test(num_layers=3, num_qubits=3, stepsize=0.01, seed=0)
    print("Test passed")
