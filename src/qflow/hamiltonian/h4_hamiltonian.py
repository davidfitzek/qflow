from typing import Tuple

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from pennylane import qchem

from qflow.hamiltonian.molecule import molecular_hamiltonian


def get_h4_hamiltonian(
    distance: float = 1.0,
    angle: float = 90.0,
) -> Tuple[qml.Hamiltonian, int, float, np.ndarray, np.ndarray]:
    """
    Calculates the Hamiltonian for the hydrogen (H4) molecule at a given distance and angle.

    Args:
        distance (float): The distance between the H atoms in angstroms. Default: 1.0.
        angle (float): The angle between adjacent H atoms in degrees. Default: 90.0.

    Returns:
        A tuple containing the following values:
        - H (qml.Hamiltonian): The molecular Hamiltonian.
        - num_qubits (int): The number of qubits required to represent the Hamiltonian.
        - min_energy (float): The minimum energy of the molecule.
        - hf_occ (np.ndarray): The Hartree-Fock occupation numbers.
        - no_occ (np.ndarray): The natural orbital occupation numbers.
    """
    name = "H4"
    symbols = ["H", "H", "H", "H"]
    active_electrons = 4
    active_orbitals = 4
    frozen = 0

    x = distance * pnp.cos(pnp.deg2rad(angle) / 2.0)
    y = distance * pnp.sin(pnp.deg2rad(angle) / 2.0)

    coordinates = pnp.array([[x, y, 0.0], [x, -y, 0.0], [-x, -y, 0.0], [-x, y, 0.0]])

    (
        H,
        num_qubits,
        min_energy,
        hf_occ,
        no_occ,
    ) = molecular_hamiltonian(
        symbols=symbols,
        coordinates=coordinates,
        name=name,
        frozen=frozen,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    return H, num_qubits, min_energy, hf_occ, no_occ
