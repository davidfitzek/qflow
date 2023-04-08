import jax.numpy as jnp
import numpy as np
import pennylane as qml
from jax import vmap

from qflow.utils.utils import get_all_bitstrings

from .abstract_initial_state import InitialState


class DickeState(InitialState):
    def __init__(self, num_qubits, hamming_weight, **kwargs):
        super().__init__(num_qubits, **kwargs)
        self.state = np.zeros(2**self.num_qubits, dtype=np.complex128)
        all_bitstrings = get_all_bitstrings(self.num_qubits)
        sum_bitstrings = vmap(jnp.sum)(all_bitstrings)  # nice!
        self.index_ = list(*jnp.where(sum_bitstrings == hamming_weight))
        self.wires = range(self.num_qubits)
        for i in self.index_:
            self.state[i] = 1 / np.sqrt(len(self.index_)) * 1 + 0j

    def __call__(self):
        qml.QubitStateVector(self.state, wires=self.wires)
