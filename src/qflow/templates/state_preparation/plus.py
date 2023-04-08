import numpy as np
import pennylane as qml

from .abstract_initial_state import InitialState


class Plus(InitialState):
    def __init__(self, num_qubits: int, **kwargs):
        super().__init__(num_qubits, **kwargs)

    def __call__(self):
        wires = range(self.num_qubits)
        for w in wires:  # Plus state
            qml.Hadamard(wires=w)
