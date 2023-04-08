import pennylane as qml
import networkx as nx


def get_heisenberg_hamiltonian(n_spins: int):
    """
    Builds a heisenberg model on a square lattice of dim (n_spins, n_spins).

    The ground state of such as system is known as the singlet state. (Ref?)



    H=∑⟨𝑖,𝑗⟩[𝑋𝑖𝑋𝑗+𝑌𝑖𝑌𝑗+𝑍𝑖𝑍𝑗]

    """
    graph = nx.generators.lattice.grid_2d_graph(n_spins, n_spins)
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    obs = []
    coeffs = []
    for u, v in graph.edges():
        coeffs.extend([1.0 for _ in range(n_spins)])
        obs.extend(
            [
                qml.PauliX(u) @ qml.PauliX(v),
                qml.PauliY(u) @ qml.PauliY(v),
                qml.PauliZ(u) @ qml.PauliZ(v),
            ]
        )
    hamiltonian_heisenberg = qml.Hamiltonian(coeffs, obs)
    return hamiltonian_heisenberg
