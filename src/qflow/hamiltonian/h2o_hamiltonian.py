from typing import Tuple

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from qflow.hamiltonian.molecule import molecular_hamiltonian


def get_h2o_hamiltonian(
    distance: float = 1.32,
    angle: float = 104.5,
) -> Tuple[qml.Hamiltonian, int, float, np.ndarray, np.ndarray]:
    """
    Calculates the Hamiltonian for the water (H2O) molecule at a given distance and angle.

    Args:
        distance (float): The distance between the O and H atoms in angstroms. Default: 1.32.
        angle (float): The H-O-H bond angle in degrees. Default: 104.5.

    Returns:
        A tuple containing the following values:
        - H (qml.Hamiltonian): The molecular Hamiltonian.
        - num_qubits (int): The number of qubits required to represent the Hamiltonian.
        - min_energy (float): The minimum energy of the molecule.
        - hf_occ (np.ndarray): The Hartree-Fock occupation numbers.
        - no_occ (np.ndarray): The natural orbital occupation numbers.
    """
    name = "h2o"
    symbols = ["O", "H", "H"]
    frozen = 1
    active_electrons = 8
    active_orbitals = 6

    x = distance * pnp.cos(pnp.deg2rad(angle) / 2.0)
    y = distance * pnp.sin(pnp.deg2rad(angle) / 2.0)

    coordinates = pnp.array([[0.0, 0.0, 0.0], [x, y, 0.0], [x, -y, 0.0]])

    (H, num_qubits, min_energy, hf_occ, no_occ) = molecular_hamiltonian(
        symbols=symbols,
        coordinates=coordinates,
        name=name,
        frozen=frozen,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
    return H, num_qubits, min_energy, hf_occ, no_occ
