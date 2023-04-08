from typing import List, Tuple

import autograd
import pennylane as qml
import pennylane.numpy as pnp
from pennylane.operation import Operation

from qflow.templates.abstract_molecular_circuit import AbstractMolecularCircuit


class MolecularBasicEntangler(AbstractMolecularCircuit):
    """A hardware efficient circuit ansatz for molecular Hamiltonians.

    This class is a subclass of `AbstractCircuit` and is used to implement a
    hardware efficient ansatz for molecular Hamiltonians. The class provides
    functions to define the structure of the circuit, the underlying parameters

    Args:
        num_layers (int, optional): The number of layers in the circuit. Defaults to 1.
        wires (List): The wires in the circuit. Defaults to None.
        initial_state (pnp.ndarray): The initial state of the circuit. Defaults to None.

    Attributes:
        num_layers (int): The number of layers in the circuit.
        H (qml.Hamiltonian): The molecular Hamiltonian.
        initial_state (pnp.ndarray): The initial state of the circuit.
        num_qubits (int): The number of qubits in the circuit.
        params_shape (List[int]): The shape of the parameters in the circuit.

    """

    def __init__(
        self,
        num_layers: int = 1,
        wires: List = None,
        initial_state: pnp.ndarray = None,
    ):
        super().__init__(num_layers, wires, initial_state)  # for parent class Optimizer
        self.params_shape = qml.BasicEntanglerLayers.shape(
            n_layers=self.num_layers, n_wires=self.num_qubits
        )

    def _circuit_ansatz(self, params: pnp.ndarray) -> Operation:
        """Perform the actual circuit evaluation with given parameters.

        Args:
            params (pnp.ndarray): The parameters to be used for evaluating the circuit.

        Returns:
            Operation: The evaluation of the circuit.

        Raises:
            AssertionError: If `params` does not have the expected shape or if
                `num_qubits` is not an integer.
        """
        params = params.reshape(self.params_shape)

        assert isinstance(self.num_qubits, int), "num_qubits is not an int"
        assert self.params_shape == params.shape, "params shape is wrong"

        wires = range(self.num_qubits)
        qml.BasisState(self.initial_state, wires=wires)
        qml.BasicEntanglerLayers(weights=params, wires=wires, rotation=qml.RY)

        # params = pnp.array(params)
        # pennylanes Adam optimizer returns a numpy_boxes.ArrayBox instead of a pnp.tensor
        # Bug or feature?
        # assert isinstance(params, pnp.tensor) or isinstance(
        # params, autograd.numpy.numpy_boxes.ArrayBox
        # ), f"params must be either a pnp.tensor or autograd.numpy.numpy_boxes.ArrayBox but got {type(params)}"
