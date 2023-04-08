from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# from beartype import beartype
import numpy as np
import pennylane as qml


def get_hamiltonian_matrix(H: qml.Hamiltonian):
    """Print the Hamiltonian."""
    return qml.utils.sparse_hamiltonian(H).toarray()


def get_all_bitstrings_as_string(n: int) -> List:
    """Return a list of strings representing all the bitstrings up to n.

    Args:
        n_qubits (int): The length of each bitstring.

    Returns:
        List: The list of all bitstrings as strings.
    """
    bitstrings = get_all_bitstrings(n)
    all_bitstrings_str = [
        " ".join(str(b) for b in bitstring) for bitstring in bitstrings
    ]
    return all_bitstrings_str


def get_approximation_ratio(E_optimized: float, E_min: float, E_max: float) -> float:
    """Calculate the approximation ratio.

    Args:
        E_optimized (float): the expectation value.
        E_min (float): The lowest energy eigenstate.
        E_max (float): The highest energy eigenstate.

    Returns:
        r: The approximation ratio.
    """
    return (E_optimized - E_max) / (E_min - E_max)


def get_all_bitstrings(n_bits: int) -> np.ndarray:
    """Get all bitstrings up to n_bits.

    Args:
        n_bits (int): The length of the bitstring.

    Returns:
        bitstrings_binary (np.ndarray): An array of shape (n_permutations, n_bits).
    """
    n_bitstrings = 2**n_bits
    bitstrings = np.zeros(shape=(n_bitstrings, n_bits), dtype=np.int8)
    tf = np.array([False, True])
    for i in range(n_bits):
        j = n_bits - i - 1
        bitstrings[np.tile(np.repeat(tf, 2**j), 2**i), i] = 1

    return bitstrings


def get_all_spinstrings(n_bits: int) -> np.ndarray:
    """Get all spinstrings up to n_bits.

    Args:
        n_bits (int): The length of the spinstring.

    Returns:
        spinstrings_binary (np.ndarray): An array of shape (n_permutations, n_bits).
    """
    bitstrings = get_all_bitstrings(n_bits)
    spinstrings = 1 - 2 * bitstrings

    return spinstrings


def pairwise(iterable: range) -> List:
    a = iter(iterable)
    return zip(a, a)
