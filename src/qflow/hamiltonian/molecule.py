from typing import List, Tuple

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from openfermion import MolecularData
from openfermionpyscf import prepare_pyscf_molecule
from pennylane import Hamiltonian
from pennylane import numpy as np
from pennylane import qchem
from pennylane.qchem.convert import import_operator
from pyscf import cc, fci, scf


def molecular_hamiltonian(
    symbols: List[str] = ["H", "H"],
    coordinates: np.ndarray = np.array([[0, 0, 0], [0, 0, 1.0]]),
    name: str = "molecule",
    charge: int = 0,
    mult: int = 1,
    basis: str = "sto-6g",
    method: str = "pyscf",
    active_electrons: int = None,
    active_orbitals: int = None,
    mapping: str = "jordan_wigner",
    wires: int = None,
    frozen: int = 0,
) -> Tuple[Hamiltonian, int, float, np.ndarray, np.ndarray]:
    """Compute the molecular Hamiltonian, the number of qubits, the exact energy, and orbital occupancies.

    Args:
    symbols: List[str], the element symbols of the molecule (default: ['H', 'H']).
    coordinates: np.ndarray, the coordinates of the molecule (default: [[0, 0, 0], [0, 0, 1.0]]).
    name: str, name of the molecule (default: 'molecule').
    charge: int, the charge of the molecule (default: 0).
    mult: int, the multiplicity of the molecule (default: 1).
    basis: str, the basis set used to calculate the energy (default: 'sto-6g').
    method: str, the method used to calculate the energy (default: 'pyscf').
    active_electrons: int, the number of active electrons (default: None).
    active_orbitals: int, the number of active orbitals (default: None).
    mapping: str, the qubit mapping used for the Hamiltonian (default: 'jordan_wigner').
    wires: int, the number of wires for the Hamiltonian (default: None).
    frozen: int, the number of orbitals to freeze (default: 0).

    Returns:
    Tuple: (Hamiltonian, number of qubits, exact energy, Hartree-Fock occupancy, natural orbital occupancy)

    """
    if len(coordinates) == len(symbols) * 3:
        geometry_hf = coordinates
    elif len(coordinates) == len(symbols):
        geometry_hf = coordinates.flatten()
    hf_file = qchem.meanfield(
        symbols, geometry_hf, name, charge, mult, basis, method, "."
    )
    molecule = MolecularData(filename=hf_file)
    # addition to get the FCI energy directly
    pyscf_molecule = prepare_pyscf_molecule(molecule)
    if pyscf_molecule.spin:
        pyscf_scf = scf.ROHF(pyscf_molecule).run()
    else:
        pyscf_scf = scf.RHF(pyscf_molecule).run()
    if frozen > 0:
        mycc = cc.CCSD(pyscf_scf, frozen=frozen).run()
        et = mycc.ccsd_t()
        fci_energy = mycc.e_tot + et
        rdm = mycc.make_rdm1()
        hf_occ = np.diag(rdm)
        no_occ = np.linalg.eigh(rdm)[0][::-1]
    else:
        pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
        fci_energy, fci_vec = pyscf_fci.kernel()
        rdm = pyscf_fci.make_rdm1(
            fci_vec, pyscf_molecule.nbas, pyscf_molecule.nelectron
        )
        hf_occ = np.diag(rdm)
        no_occ = np.linalg.eigh(rdm)[0][::-1]
    core, active = qchem.active_space(
        molecule.n_electrons,
        molecule.n_orbitals,
        mult,
        active_electrons,
        active_orbitals,
    )
    # openfermion version of hamiltonian
    h_of, qubits = (
        qchem.decompose(hf_file, mapping, core, active),
        2 * len(active),
    )

    h_pl = import_operator(h_of, wires=wires)
    H = Hamiltonian(h_pl.coeffs, h_pl.ops)
    return H, qubits, fci_energy, hf_occ, no_occ
