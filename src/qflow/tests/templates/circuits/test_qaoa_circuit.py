import numpy as np
import pennylane as qml
import pytest
from jax import jit

from qflow.qaoa.mixer_h import circular_xy_mixer, row_mixer, row_mixer_2, x_mixer
from qflow.templates.circuits import QAOACircuit
from qflow.templates.state_preparation import Plus


def H(N):
    def get_qubit_index(i: int, j: int, N: int):
        return i * N + j

    coeffs = [(2 - N) for v in range(N) for j in range(N)]
    obs = [qml.PauliZ(get_qubit_index(v, j, N)) for v in range(N) for j in range(N)]
    h_z = qml.Hamiltonian(coeffs, obs)

    coeffs = [1 for v in range(N) for j in range(N) for j_ in range(N) if j < j_]
    obs = [
        qml.PauliZ(get_qubit_index(v, j, N)) @ qml.PauliZ(get_qubit_index(v, j_, N))
        for v in range(N)
        for j in range(N)
        for j_ in range(N)
        if j < j_
    ]
    h_zz = qml.Hamiltonian(coeffs, obs)

    return h_zz + h_z


@pytest.mark.parametrize(
    "H, initial_state, mixer_h, num_layers",
    [
        (H(2), Plus, x_mixer, 1),
    ],
)
def test_circuit(
    H: qml.qnode, initial_state: qml.qnode, mixer_h: qml.qnode, num_layers: int
):
    seed = 0
    wires = H.wires
    num_qubits = len(wires)
    mixer_h = mixer_h(num_qubits)
    initial_state = Plus(num_qubits)
    assert mixer_h.wires == H.wires, f"mixer: {mixer_h.wires}, cost: {H.wires}"

    circ = QAOACircuit(
        H=H, initial_state=initial_state, mixer_h=mixer_h, num_layers=num_layers
    )
    init_params = circ.init(seed)

    assert circ.num_qubits == num_qubits
    assert circ.num_layers == num_layers


if __name__ == "__main__":
    test_circuit(H(2), Plus, circular_xy_mixer, 1)
