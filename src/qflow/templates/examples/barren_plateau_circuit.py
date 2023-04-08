import numpy as np

from qflow.hamiltonian.utils_hamiltonian import get_eigenvalues_hamiltonian
from qflow.templates.circuits import BarrenPlateauCircuit


def barren_plateau_example(
    num_layers: int, num_qubits: int = 4
) -> BarrenPlateauCircuit:
    """Return a BarrenPlateauCircuit instance with the specified number of qubits and layers.
        https://arxiv.org/pdf/1803.11173.pdf

    Args:
        num_qubits (int): The number of qubits to use in the circuit.
        num_layers (int): The number of layers in the circuit.

    Returns:
        BarrenPlateauCircuit: An instance of the BarrenPlateauCircuit class with the specified number of qubits and layers.
        H (qml.Hamiltonian): The Hamiltonian of the MaxCut problem.
        min_energy (float): The minimum energy of the MaxCut problem.

    Example:
    >>> circuit, H, min_energy = barren_plateau_circuit(num_qubits=2, num_layers=3)
    >>> print(circuit.num_qubits)
    2
    >>> print(circuit.num_layers)
    3
    >>> params = circuit.init()
    >>> print(params.shape)
    (6,)

    """

    circuit = BarrenPlateauCircuit(num_layers=num_layers, num_qubits=num_qubits)
    eigenvalues = get_eigenvalues_hamiltonian(circuit.H)
    min_energy = np.min(eigenvalues)

    return circuit, circuit.H, min_energy
