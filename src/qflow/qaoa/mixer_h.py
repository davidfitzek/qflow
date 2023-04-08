import networkx as nx
import numpy as np
import pennylane as qml

# from beartype import beartype
from pennylane import qaoa

from qflow.utils.utils import pairwise


def circular_xy_mixer(n_qubits: int) -> qml.Hamiltonian:
    """
    Create an XY mixer for a circular graph with n_qubits number of nodes.

    Args:
        n_qubits (int): number of qubits

    Returns:
        mixer_h: XY mixer Hamiltonian
    """
    list_1 = [(u, v) for u, v in pairwise(range(n_qubits))]
    list_2 = [(u, v) for u, v in pairwise(np.roll(range(n_qubits), -1))]
    graph = nx.Graph(list_1 + list_2)
    mixer_h = qaoa.xy_mixer(graph)
    return mixer_h


def row_mixer(n_qubits: int) -> qml.Hamiltonian:
    """
    Create an XY mixer for a 1-dimensional row graph with n_qubits number of nodes.

    Args:
        n_qubits (int): number of qubits

    Returns:
        mixer_h: XY mixer Hamiltonian
    """

    N = int(np.sqrt(n_qubits))
    list_ = [(i + j * N, (i + 1) % N + j * N) for i in range(N) for j in range(N)]
    graph = nx.Graph(list_)
    mixer_h = qaoa.xy_mixer(graph)
    return mixer_h


def row_mixer_2(n_qubits: int) -> qml.Hamiltonian:
    """
    Create an XY mixer for a 2-dimensional row graph with n_qubits number of nodes.

    Args:
        n_qubits (int): number of qubits

    Returns:
        mixer_h: XY mixer Hamiltonian
    """

    N = int(np.sqrt(n_qubits))
    list_ = [(i + j * N, (i + 1) % N + j * N) for i in range(N - 1) for j in range(N)]
    graph = nx.Graph(list_)
    mixer_h = qaoa.xy_mixer(graph)
    return mixer_h


def row_flex_mixer(n_qubits: int) -> list[qml.Hamiltonian]:
    """
    Create an XY mixer for a flexible row graph with n_qubits number of nodes.

    Args:
        n_qubits (int): number of qubits

    Returns:
        mixer_h_list: list of XY mixer Hamiltonians
    """

    N = int(np.sqrt(n_qubits))
    list_ = [[(i + j * N, (i + 1) % N + j * N) for i in range(N - 1)] for j in range(N)]
    mixer_h_list = []
    for sub_list in list_:
        graph = nx.Graph(sub_list)
        mixer_h_list.append(qaoa.xy_mixer(graph))
    return mixer_h_list


def x_mixer(n_qubits: int) -> qml.Hamiltonian:
    """
    Create an X mixer for a 1-dimensional row graph with n_qubits number of nodes.

    Args:
        n_qubits (int): number of qubits

    Returns:
        mixer_h: X mixer Hamiltonian
    """
    mixer_h = qaoa.x_mixer(range(n_qubits))
    return mixer_h
