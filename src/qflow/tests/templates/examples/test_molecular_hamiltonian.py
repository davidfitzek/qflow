from copy import deepcopy
from typing import Callable

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest

# from jax import jit
# from jaxopt import OptaxSolver
from pennylane import AdamOptimizer, QNGOptimizer, qchem
from tqdm import tqdm

from qflow.hamiltonian.molecule import molecular_hamiltonian
from qflow.templates.examples import (
    h2o_vqe_basic_entangler_example,
    h2o_vqe_strong_entangler_example,
    h4_vqe_basic_entangler_example,
    h4_vqe_strong_entangler_example,
    lih_vqe_basic_entangler_example,
    lih_vqe_strong_entangler_example,
)
from qflow.utils.utils import get_approximation_ratio


@pytest.mark.parametrize(
    "get_example, num_layers, seed",
    [
        (h2o_vqe_basic_entangler_example, 2, 0),
        (h4_vqe_basic_entangler_example, 3, 0),
        (lih_vqe_basic_entangler_example, 2, 0),
    ],
)
def test_several_molecular_hamiltonian(get_example, num_layers, seed):
    circuit, H, min_energy = get_example(num_layers)
    params = circuit.init(seed=seed)

    dev = qml.device("default.qubit", wires=len(circuit.wires))

    @qml.qnode(dev)
    def loss_fn(params):
        circuit(params)
        return qml.expval(H)

    optimizer = AdamOptimizer(0.05)
    for _ in tqdm(range(100)):
        params = optimizer.step(loss_fn, params)

    min_cost_optimized = loss_fn(params)
    r = get_approximation_ratio(min_cost_optimized, min_energy, 0)

    assert (
        r > 0.5
    ), f"Optimization failed for {circuit} with an approximation ratio of {np.round(r,3)}, the optimized energy is: {np.round(min_cost_optimized,3)}, the ground state energy: {np.round(min_energy,3)}"

    print(f"ground state: {np.round(min_energy, 1)}")
    print(f"exp val: {np.round(min_cost_optimized, 2)}")
    print(f"approx. ratio: {np.round(r,2).real}")


def test_h4_one_layer_convergence():
    circuit, H, min_energy = h4_vqe_strong_entangler_example(num_layers=1)
    # print("np.array(" + np.array2string(params, separator=", ") + ")")
    params = pnp.array(
        [
            [
                [-0.52136612, 1.38433962, -3.14087402],
                [-1.24198108, -2.2194982, -2.56141215],
                [-1.97128523, -0.97037057, -0.64862909],
                [0.24389273, -0.50771584, 1.16376844],
                [-1.85698128, 2.37578192, -2.96951133],
                [1.07107896, -0.51958925, 0.36875907],
                [-2.2595155, -1.89688429, 1.88963386],
                [2.94217425, -1.17229046, 1.20839863],
            ]
        ]
    )  # these are the params from the seed = 1
    # params = circuit.init(seed=2)
    # print(params)
    # print("np.array(" + np.array2string(params, separator=", ") + ")")

    # optimizer = AdamOptimizer(0.05)
    optimizer = QNGOptimizer(0.05, approx="block-diag", lam=1e-7)

    dev = qml.device(
        "default.qubit", wires=range(circuit.num_qubits + 1)
    )  # add one aux qubit for calculating the full Fisher matrix.

    @qml.qnode(dev)
    def get_expval(params):
        circuit(params)
        return qml.expval(H)

    print(f"init expval: {get_expval(params)}")

    for i in range(800):
        # assert that params is instance of pnp.tensor.tensor
        assert isinstance(
            params, pnp.tensor
        ), f"Params must be of type pnp.tensor, but got {type(params)}"

        prev_params = params
        params, exp_val = optimizer.step_and_cost(get_expval, params)

        assert isinstance(
            params, pnp.tensor
        ), f"Params must be of type pnp.tensor, but got {type(params)}"

        diff = np.sum(np.abs(prev_params - params))

        if i % 50 == 0:
            print(f"step: {i}, exp val: {np.round(exp_val, 2)}")
            print(f"diff: {diff}")

        if exp_val < -1.624 or diff < 1e-4:
            print(f"converged after {i} steps")
            break

    assert np.isclose(
        exp_val, -1.624, atol=0.02
    ), f"Circuit did not converge with exp val: {np.round(exp_val,3)}"
    # Adam convergence after 70 steps
    # QNG convergence after 74 steps


def test_multiple_inits():
    circuit, H, min_energy = h4_vqe_strong_entangler_example(num_layers=1)
    for i in range(3):
        params = circuit.init(seed=i)
        print(params)

        # assert that all params are different
        assert not np.allclose(params, circuit.init(seed=i + 1))


if __name__ == "__main__":
    # pytest libs/qflow/src/qflow/tests/templates/examples/test_molecular_hamiltonian.py

    # test_h4_one_layer_convergence()
    # test_several_molecular_hamiltonian(h4_vqe_basic_entangler_example, 3, 0)
    test_several_molecular_hamiltonian(h2o_vqe_basic_entangler_example, 2, 0)
    test_several_molecular_hamiltonian(lih_vqe_basic_entangler_example, 3, 0)
