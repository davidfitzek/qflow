import pennylane.numpy as pnp
from pennylane import qchem

from qflow.hamiltonian.molecule import molecular_hamiltonian


def test_h4_hamiltonian():
    name = "H4"
    symbols = ["H", "H", "H", "H"]
    active_electrons = 4
    active_orbitals = 4
    frozen = 0
    distance = 1.0
    angle = 90.0
    x = distance * pnp.cos(pnp.deg2rad(angle) / 2.0)
    y = distance * pnp.sin(pnp.deg2rad(angle) / 2.0)

    coordinates = pnp.array([[x, y, 0.0], [x, -y, 0.0], [-x, -y, 0.0], [-x, y, 0.0]])
    mult = 1
    charge = 0
    basis = "sto-6g"

    H_test, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=charge,
        mult=mult,
        basis=basis,
        active_electrons=4,
        active_orbitals=4,
        method="pyscf",
    )

    # H4 Hamiltonian
    (
        H,
        num_qubits,
        min_energy,
        hf_occ,
        no_occ,
    ) = molecular_hamiltonian(
        symbols=symbols,
        coordinates=coordinates,
        name=name,
        frozen=frozen,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    for term_1, term_2 in zip(H.ops, H_test.ops):
        assert str(term_1) == str(term_2), "The terms are not equal"

    print("done")


if __name__ == "__main__":
    test_h4_hamiltonian()
