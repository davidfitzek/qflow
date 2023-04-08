from typing import Tuple

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from qflow.hamiltonian.molecule import molecular_hamiltonian


def get_lih_hamiltonian(
    distance: float = 3.0,
    frozen: int = 1,
    active_electrons: int = 2,
    active_orbitals: int = 5,
) -> Tuple[qml.Hamiltonian, int, float, np.ndarray, np.ndarray]:
    """
    Calculates the Hamiltonian for the lithium hydride (LiH) molecule at a given distance.

    Args:
        distance (float): The distance between the Li and H atoms in angstroms. Default: 3.0.
        frozen (int): The number of frozen molecular orbitals. Default: 1.
        active_electrons (int): The number of active electrons in the molecule. Default: 2.
        active_orbitals (int): The number of active molecular orbitals. Default: 5.

    Returns:
        A tuple containing the following values:
        - H (qml.Hamiltonian): The molecular Hamiltonian.
        - num_qubits (int): The number of qubits required to represent the Hamiltonian.
        - min_energy (float): The minimum energy of the molecule.
        - hf_occ (np.ndarray): The Hartree-Fock occupation numbers.
        - no_occ (np.ndarray): The natural orbital occupation numbers.
    """
    name = "LiH"
    symbols = ["Li", "H"]
    coordinates = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])

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
