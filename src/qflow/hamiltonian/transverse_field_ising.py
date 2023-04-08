import numpy as np
import pennylane as qml


def get_transverse_field_ising_hamiltonian(
    n_spins: int, transverse_field: float
) -> qml.Hamiltonian:
    """
    Calculates the Hamiltonian of the transverse field Ising model.

    The transverse field Ising model is a simplified model of interacting spins used to study quantum phase transitions and quantum magnetism. The model consists of a chain of N spins, each coupled to its nearest neighbor with a ferromagnetic interaction strength J and subject to a transverse magnetic field B.

    Args:
        n_spins (int): The number of spins in the chain.
        transverse_field (float): The strength of the transverse magnetic field.

    Returns:
        qml.Hamiltonian: The Hamiltonian of the transverse field Ising model.

    Raises:
        ValueError: If the number of spins is not a positive integer.
        ValueError: If the strength of the transverse field is negative.
    """
    if not isinstance(n_spins, int) or n_spins <= 0:
        raise ValueError("Number of spins must be a positive integer.")

    if transverse_field < 0:
        raise ValueError("Transverse field strength must be non-negative.")

    # transverse field
    coeffs = [-transverse_field for _ in range(n_spins)]
    obs = [qml.PauliX(np.pi / 2) for _ in range(n_spins)]
    h_x = qml.Hamiltonian(coeffs, obs)

    # spin chain
    couplings = [(i, (i + 1) % n_spins) for i in range(n_spins)]
    coeffs = [-1 for _ in range(n_spins)]
    obs = [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in couplings]
    h_zz = qml.Hamiltonian(coeffs, obs)

    return h_zz + h_x
