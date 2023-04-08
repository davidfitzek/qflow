import networkx as nx
import pennylane as qml


def get_maxcut_hamiltonian(graph: nx.Graph):
    """
    Calculates the Maxcut Hamiltonian for a given graph.

    The Maxcut problem is the optimization problem of finding the partition of
    the nodes in a graph that maximizes the number of edges crossing the two partitions. The Maxcut Hamiltonian is a quantum Hamiltonian that encodes this optimization problem and is used in quantum algorithms to solve it.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        qml.Hamiltonian: The Maxcut Hamiltonian for the given graph.

    Raises:
        TypeError: If the input graph is not a networkx Graph class instance.
    """

    if not isinstance(graph, nx.Graph):
        raise TypeError(
            "Graph is not a networkx class (found type: %s)" % type(graph).__name__
        )

    coeffs = [-0.5 for _ in graph.edges]
    obs = [qml.Identity(e[0]) @ qml.Identity(e[1]) for e in graph.edges]
    identity_h = qml.Hamiltonian(coeffs, obs)

    coeffs = [0.5 for (_, _) in graph.edges()]
    obs = [qml.PauliZ(u) @ qml.PauliZ(v) for (u, v) in graph.edges()]
    zz_h = qml.Hamiltonian(coeffs, obs)

    return zz_h + identity_h
