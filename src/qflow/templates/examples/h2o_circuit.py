import pennylane.numpy as pnp

from qflow.hamiltonian.h2o_hamiltonian import get_h2o_hamiltonian
from qflow.templates.circuits import (
    AbstractMolecularCircuit,
    MolecularBasicEntangler,
    MolecularStrongEntangler,
)


def h2o_vqe_basic_entangler_example(
    num_layers: int = 1,
    distance: float = 1.32,
    angle: float = 104.5,
) -> AbstractMolecularCircuit:
    """
    Creates a VQE circuit for the H2O molecule using the Basic Entangler Layers ansatz.

    The Basic Entangler Layers ansatz is a parametrized quantum circuit that can
    be used to prepare trial states for VQE.

    Args:
        num_layers (int): The number of layers in the ansatz. Default: 1.
        distance (float): The O-H bond distance in the H2O molecule, in angstroms. Default: 1.32.
        angle (float): The H-O-H bond angle in degrees. Default: 104.5.
        seed (int): The random seed for the Gaussian noise. Default: 0.

    Returns:
        A tuple containing the following values:
        - circuit (AbstractMolecularCircuit): The Basic Entangler Layers ansatz circuit.
        - H (qml.Hamiltonian): The molecular Hamiltonian for the H2O molecule.
        - min_energy (float): The minimum energy of the molecule.

    Example:
    >>> circuit, H, min_energy = h2o_vqe_basic_entangler_example(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """

    H, num_qubits, min_energy, hf_occ, no_occ = get_h2o_hamiltonian(distance, angle)

    initial_state = pnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

    circuit = MolecularBasicEntangler(
        num_layers=num_layers,
        wires=H.wires,
        initial_state=initial_state,
    )

    return circuit, H, min_energy


def h2o_vqe_strong_entangler_example(
    num_layers: int = 1,
    distance: float = 1.32,
    angle: float = 104.5,
) -> AbstractMolecularCircuit:
    """
    Build a quantum circuit for the Variational Quantum Eigensolver (VQE)
        simulation of the H2O molecule.

    Args:
        num_layers (int, optional): Number of layers in the circuit. Defaults to 1.
        distance (float, optional): Distance between atoms in the molecule.
            Defaults to 1.0.
        angle (float, optional): Angle between atoms in the molecule.
            Defaults to 90.0.
        hf_params (bool, optional): Flag to specify whether to use Hartree-Fock parameters.
            Defaults to False.
        perturb_hf_params (bool, optional): Flag to specify whether to perturb Hartree-Fock parameters.
            Defaults to True.
        seed (int, optional): Seed for random number generation. Defaults to 0.

    Returns:
        MolecularHamiltonianCircuit: A quantum circuit object representing the H2O molecule.
        H: The Hamiltonian of the H2O molecule.
        min_energy: The ground state energy of the H2O molecule.


    Example:
    >>> circuit, H, min_energy = h2o_vqe_circuit(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """
    H, num_qubits, min_energy, hf_occ, no_occ = get_h2o_hamiltonian(distance, angle)

    initial_state = pnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

    circuit = MolecularStrongEntangler(
        num_layers=num_layers,
        wires=H.wires,
        initial_state=initial_state,
    )

    return circuit, H, min_energy
