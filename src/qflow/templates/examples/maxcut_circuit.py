import numpy as np
import pennylane as qml

from qflow.hamiltonian.utils_hamiltonian import get_eigenvalues_hamiltonian
from qflow.qaoa.mixer_h import x_mixer
from qflow.templates.circuits import QAOACircuit
from qflow.templates.state_preparation import Plus
from qflow.utils.maxcut_utils import get_maxcut_graph


def maxcut_qaoa_example(
    num_layers: int = 1, num_nodes: int = 4, seed: int = 0
) -> QAOACircuit:
    """Generate a QAOA circuit for the MaxCut problem.

    Args:
        num_layers (int, optional): The number of layers of the QAOA circuit.
            Defaults to 1.
        num_nodes (int, optional): The number of nodes in the MaxCut graph.
            Defaults to 4.
        seed (int, optional): Random seed. Defaults to 1.

    Returns:
        circuit (QAOACircuit): The quantum circuit class.
        H (qml.Hamiltonian): The Hamiltonian of the MaxCut problem.
        min_energy (float): The minimum energy of the MaxCut problem.

    Example:
    >>> circuit, H, min_energy = maxcut_qaoa_circuit(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """

    graph = get_maxcut_graph(num_nodes, seed=seed)
    H, _ = qml.qaoa.maxcut(graph)

    num_qubits = len(H.wires)
    mixer_h = x_mixer(num_qubits)
    initial_state = Plus(num_qubits)

    circuit = QAOACircuit(
        H=H, initial_state=initial_state, mixer_h=mixer_h, num_layers=num_layers
    )

    eigenvalues = get_eigenvalues_hamiltonian(H)
    min_energy = np.min(eigenvalues)

    return circuit, H, min_energy
