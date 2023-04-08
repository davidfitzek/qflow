import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest

from qflow.hamiltonian.utils_hamiltonian import get_eigenvalues_hamiltonian
from qflow.templates.circuits import BarrenPlateauCircuit


@pytest.mark.parametrize(
    "num_layers, num_qubits, seed",
    [
        (5, 7, 0),
    ],
)
def test_diff_method(num_layers, num_qubits, seed):
    # circuit
    circuit = BarrenPlateauCircuit(num_layers, num_qubits)
    params = circuit.init(seed)
    # Hamiltonian
    eigenvalues = get_eigenvalues_hamiltonian(circuit.H)
    min_cost, max_cost = np.min(eigenvalues), np.max(eigenvalues)

    dev = qml.device("default.qubit", wires=circuit.wires)

    @qml.qnode(dev, diff_method="parameter-shift")
    def fun_parameter_shift(params):
        circuit(params)
        return qml.expval(circuit.H)

    dev = qml.device("default.qubit", wires=circuit.wires)

    @qml.qnode(dev)
    def fun(params):
        circuit(params)
        return qml.expval(circuit.H)

    for _ in range(50):
        params = pnp.random.rand(
            35,
        ) * pnp.random.rand(
            35,
        )
        gradients = qml.grad(fun)(params)
        gradients_parameter_shift = qml.grad(fun_parameter_shift)(params)
        np.testing.assert_almost_equal(gradients, gradients_parameter_shift, decimal=7)


if __name__ == "__main__":
    test_diff_method(5, 7, 0)
