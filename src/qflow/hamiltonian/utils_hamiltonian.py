from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml
from scipy.linalg import eig, inv


def get_eigenvalues_hamiltonian(H: qml.Hamiltonian) -> np.ndarray:
    """Get the eigenvalues of the cost hamiltonian.

    Args:
        hamiltonian (qml.Hamiltonian): The cost hamiltonian.

    Returns:
        eigenvalues (np.ndarray): An array with the eigenvalues of the cost hamiltonian.
    """
    Hmat = qml.utils.sparse_hamiltonian(H)
    Hmat = Hmat.toarray()
    eigenvalues, eigenvectors = eig(Hmat)
    return eigenvalues


def get_eigenvalues_hermitian(H: qml.Hamiltonian) -> np.ndarray:
    """Get the eigenvalues of an Hermitian observable.

    Args:
        hermitian (qml.Hamiltonian): The hermitian observable.

    Returns:
        eigenvalues (np.ndarray): An array with the eigenvalues of the cost hamiltonian.
    """
    Hmat = H.matrix()
    eigenvalues, eigenvectors = eig(Hmat)
    return eigenvalues
