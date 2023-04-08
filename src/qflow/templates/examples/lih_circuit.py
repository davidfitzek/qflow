import pennylane.numpy as pnp

from qflow.hamiltonian.lih_hamiltonian import get_lih_hamiltonian
from qflow.hamiltonian.molecule import molecular_hamiltonian
from qflow.templates.circuits import (
    AbstractMolecularCircuit,
    MolecularBasicEntangler,
    MolecularStrongEntangler,
)


def lih_vqe_basic_entangler_example(
    num_layers: int = 1,
    distance: float = 3.0,
) -> AbstractMolecularCircuit:
    """
    Creates a VQE circuit for the LiH molecule using the Basic Entangler Layers ansatz.

    The Basic Entangler Layers ansatz is a parametrized quantum circuit that can be
    used to prepare trial states for VQE.

    Args:
        num_layers (int): The number of layers in the ansatz. Default: 1.
        distance (float): The distance between the Li and H atoms in the LiH molecule, in angstroms. Default: 3.0.

    Returns:
        A tuple containing the following values:
        - circuit (AbstractMolecularCircuit): The Basic Entangler Layers ansatz circuit.
        - H (qml.Hamiltonian): The molecular Hamiltonian for the LiH molecule.
        - min_energy (float): The minimum energy of the molecule.

    Example:
    >>> circuit, H, min_energy = lih_vqe_basic_entangler_example(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """

    H, num_qubits, min_energy, hf_occ, no_occ = get_lih_hamiltonian(distance)

    initial_state = pnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    circuit = MolecularBasicEntangler(
        num_layers=num_layers,
        wires=H.wires,
        initial_state=initial_state,
    )

    return circuit, H, min_energy


def lih_vqe_strong_entangler_example(
    num_layers: int = 1,
    distance: float = 3.0,
) -> AbstractMolecularCircuit:
    """
    Creates a VQE circuit for the LiH molecule using the Strongly Entangling Layers ansatz.

    The Strongly Entangling Layers ansatz is a parametrized quantum circuit that
    can be used to prepare trial states for VQE.

    Args:
        num_layers (int): The number of layers in the ansatz. Default: 1.
        hf_params (bool): Whether to use the Hartree-Fock parameters as initial parameters. Default: False.
        distance (float): The distance between the Li and H atoms in the LiH molecule, in angstroms. Default: 3.0.
        perturb_hf_params (bool): Whether to perturb the Hartree-Fock parameters when using them as initial parameters. Default: True.
        seed (int): The random seed for the Gaussian noise. Default: 0.

    Returns:
        A tuple containing the following values:
        - circuit (AbstractMolecularCircuit): The Strongly Entangling Layers ansatz circuit.
        - H (qml.Hamiltonian): The molecular Hamiltonian for the LiH molecule.
        - min_energy (float): The minimum energy of the molecule.

    Example:
    >>> circuit, H, min_energy = lih_vqe_strong_entangler_example(num_layers=2, hf_params=True)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """
    H, num_qubits, min_energy, hf_occ, no_occ = get_lih_hamiltonian(distance)

    initial_state = pnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    circuit = MolecularStrongEntangler(
        num_layers=num_layers,
        wires=H.wires,
        initial_state=initial_state,
    )

    return circuit, H, min_energy
