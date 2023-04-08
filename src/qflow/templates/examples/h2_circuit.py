import pennylane.numpy as pnp

from qflow.hamiltonian.molecule import molecular_hamiltonian
from qflow.templates.circuits import MolecularHamiltonianCircuit


def h2_vqe_example(
    num_layers: int = 1,
    distance: float = 1.32,
    hf_params: bool = False,
    perturb_hf_params: bool = True,
    seed: int = 0,
) -> MolecularHamiltonianCircuit:
    """
    Build a quantum circuit for the Variational Quantum Eigensolver (VQE) simulation of the H2 molecule.

    Args:
        num_layers (int, optional): Number of layers in the circuit. Defaults to 1.
        distance (float, optional): Distance between atoms in the molecule. Defaults to 1.0.
        hf_params (bool, optional): Flag to specify whether to use Hartree-Fock parameters. Defaults to False.
        perturb_hf_params (bool, optional): Flag to specify whether to perturb Hartree-Fock parameters. Defaults to True.
        seed (int, optional): Seed for random number generation. Defaults to 0.

    Returns:
        MolecularHamiltonianCircuit: A quantum circuit object representing the H2 molecule.
        H: The Hamiltonian of the H2 molecule.
        min_energy: The ground state energy of the H2 molecule.


    Example:
    >>> circuit, H, min_energy = h2_vqe_circuit(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """

    name = "h2"
    symbols = ["H", "H"]
    frozen = 0
    coordinates = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    initial_state = pnp.array([1, 1, 0, 0])
    active_electrons = 2
    active_orbitals = 2

    (H, num_qubits, min_energy, hf_occ, no_occ) = molecular_hamiltonian(
        symbols=symbols,
        coordinates=coordinates,
        name=name,
        frozen=frozen,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    circuit = MolecularHamiltonianCircuit(
        num_layers=num_layers,
        wires=H.wires,
        initial_state=initial_state,
    )

    return circuit, H, min_energy
