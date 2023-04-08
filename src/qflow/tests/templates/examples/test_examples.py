import numpy as np
import pennylane as qml
import pytest

from qflow.templates.examples import (
    h2o_vqe_basic_entangler_example,
    h4_vqe_basic_entangler_example,
    lih_vqe_basic_entangler_example,
    maxcut_qaoa_example,
)


@pytest.mark.parametrize(
    "num_layers, seed",
    [
        (2, 0),
        (1, 0),
    ],
)
def test_h2o_vqe_circuit(num_layers, seed):
    # circuit
    circuit, H, min_energy = h2o_vqe_basic_entangler_example(num_layers)
    params = circuit.init(seed)

    dev = qml.device("default.qubit", wires=circuit.wires)

    @qml.qnode(dev)
    def fun(params):
        circuit(params)
        return qml.expval(H)

    exp_val = fun(params)


@pytest.mark.parametrize(
    "num_layers, seed",
    [
        (2, 0),
        (1, 0),
    ],
)
def test_h4_vqe_circuit(num_layers, seed):
    # circuit
    circuit, H, min_energy = h4_vqe_basic_entangler_example(num_layers=num_layers)
    params = circuit.init(seed)

    dev = qml.device("default.qubit", wires=circuit.wires)

    @qml.qnode(dev)
    def fun(params):
        circuit(params)
        return qml.expval(H)

    exp_val = fun(params)


@pytest.mark.parametrize(
    "num_layers, seed",
    [
        (2, 0),
        (1, 0),
    ],
)
def test_lih_vqe_circuit(num_layers, seed):
    # circuit
    circuit, H, min_energy = lih_vqe_basic_entangler_example(num_layers)
    params = circuit.init(seed)

    dev = qml.device("default.qubit", wires=circuit.wires)

    @qml.qnode(dev)
    def fun(params):
        circuit(params)
        return qml.expval(H)

    exp_val = fun(params)


@pytest.mark.parametrize(
    "num_layers, num_nodes, seed",
    [
        (2, 5, 0),
        (1, 4, 0),
    ],
)
def test_maxcut_qaoa_circuit(num_layers, num_nodes, seed):
    # circuit
    circuit, H, min_energy = maxcut_qaoa_example(num_layers, num_nodes, seed=seed)
    params = circuit.init(seed)

    dev = qml.device("default.qubit", wires=circuit.wires)

    @qml.qnode(dev)
    def fun(params):
        circuit(params)
        return qml.expval(H)

    exp_val = fun(params)


if __name__ == "__main__":
    test_h2o_vqe_circuit(num_layers=1, seed=0)
    test_h4_vqe_circuit(num_layers=1, seed=0)
    test_lih_vqe_circuit(num_layers=2, seed=0)
    test_maxcut_qaoa_circuit(2, 4, 0)
    print("All tests passed!")
