import numpy as np
import pennylane as qml
from pennylane import numpy as qnp

from qflow.templates.circuits import AbstractCircuit


class TemplateCircuit(AbstractCircuit):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits

    @property
    def wires(self):
        return range(self.n_qubits)

    def init(self, rng_key=None):
        return self._params(rng_key)

    def _params(self, rng_key):
        return qnp.random.random((self.n_qubits)) * 2 * np.pi

    @property
    def H(self):
        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        coeffs = [1.0]
        H = qml.Hamiltonian(coeffs, obs)
        return H

    def __call__(self, x):
        return self._circuit_ansatz(x)

    def _circuit_ansatz(self, x):
        qml.RY(np.pi / 4, wires=0)
