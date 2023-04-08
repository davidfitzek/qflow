from typing import Tuple

import pennylane.numpy as pnp

from qflow.hamiltonian import get_h4_hamiltonian
from qflow.templates.circuits import (
    AbstractMolecularCircuit,
    MolecularBasicEntangler,
    MolecularStrongEntangler,
)


def h4_vqe_strong_entangler_example(
    num_layers: int = 1,
    distance: float = 1.0,
    angle: float = 90.0,
) -> AbstractMolecularCircuit:
    """
    Creates a VQE circuit for the H4 molecule using the Strongly Entangling Layers ansatz.

    The Strongly Entangling Layers ansatz is a parametrized quantum circuit that
    can be used to prepare trial states for VQE. In this function, we use the ansatz to create a circuit for the H4 molecule and calculate its ground state energy using VQE.

    Args:
        num_layers (int): The number of layers in the ansatz. Default: 1.
        distance (float): The distance between the H atoms in the H4 molecule, in angstroms. Default: 1.0.
        angle (float): The H-H-H bond angle in degrees. Default: 90.0.
        seed (int): The random seed for the Gaussian noise. Default: 0.

    Returns:
        A tuple containing the following values:
        - circuit (AbstractMolecularCircuit): The Strongly Entangling Layers ansatz circuit.
        - H (qml.Hamiltonian): The molecular Hamiltonian for the H4 molecule.
        - min_energy (float): The minimum energy of the molecule.

    Example:
    >>> circuit, H, min_energy = h4_vqe_strong_entangler_example(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """
    H, num_qubits, min_energy, hf_occ, no_occ = get_h4_hamiltonian(distance, angle)

    initial_state = pnp.array([1, 1, 1, 1, 0, 0, 0, 0])

    circuit = MolecularStrongEntangler(
        num_layers=num_layers,
        wires=H.wires,
        initial_state=initial_state,
    )

    return circuit, H, min_energy


def h4_vqe_basic_entangler_example(
    num_layers: int = 1,
    distance: float = 1.0,
    angle: float = 90.0,
) -> AbstractMolecularCircuit:
    """
    Creates a VQE circuit for the H4 molecule using the Basic Entangler Layers ansatz.

    The Basic Entangler Layers ansatz is a parametrized quantum circuit that can
    be used to prepare trial states for VQE.

    Args:
        num_layers (int): The number of layers in the ansatz. Default: 1.
        distance (float): The distance between the H atoms in the H4 molecule, in angstroms. Default: 1.0.
        angle (float): The H-H-H bond angle in degrees. Default: 90.0.
        hf_params (bool): If True, use the Hartree-Fock parameters as initial parameters for the ansatz. Default: False.
        perturb_hf_params (bool): If True, perturb the Hartree-Fock parameters by adding random Gaussian noise. Default: True.
        seed (int): The random seed for the Gaussian noise. Default: 0.

    Returns:
        A tuple containing the following values:
        - circuit (AbstractMolecularCircuit): The Basic Entangler Layers ansatz circuit.
        - H (qml.Hamiltonian): The molecular Hamiltonian for the H4 molecule.
        - min_energy (float): The minimum energy of the molecule.

    Example:
    >>> circuit, H, min_energy = h4_vqe_basic_entangler_example(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """

    H, num_qubits, min_energy, hf_occ, no_occ = get_h4_hamiltonian(distance, angle)

    initial_state = pnp.array([1, 1, 1, 1, 0, 0, 0, 0])

    circuit = MolecularBasicEntangler(
        num_layers=num_layers,
        wires=H.wires,
        initial_state=initial_state,
    )

    return circuit, H, min_energy
