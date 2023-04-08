from abc import ABC, abstractmethod
from typing import List, Tuple

import autograd
import pennylane as qml
import pennylane.numpy as pnp
from pennylane.operation import Operation
from pennylane.templates.layers import StronglyEntanglingLayers

from qflow.hamiltonian.molecule import molecular_hamiltonian
from qflow.templates.circuits import AbstractCircuit


class AbstractMolecularCircuit(AbstractCircuit):
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
        self.num_layers: int = num_layers
        self.initial_state: pnp.ndarray = initial_state
        self.num_qubits = len(wires)
        # self.HF_params = False

        self._wires = wires
        self._params = None

    @property
    def wires(self):
        """Return the wires of the circuit.

        Returns:
            Union[int, Tuple[int]]: The wires of the circuit.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        return self._wires

    def init(self, seed: int = None):
        """Initialize the parameters of the circuit.

        Args:
            seed (int, optional): The random number generator seed.
                Defaults to None.

        Returns:
            pnp.ndarray: The initialized parameters of the circuit.
        """
        if seed is not None:
            pnp.random.seed(seed)

        return self.params

    @property
    def params(self):
        """Initialize the parameters for the circuit evaluation.

        Args:
            seed (int, optional): Seed for the random number generator.

        Returns:
            pnp.ndarray: The parameters for the circuit evaluation.
        """
        self._params = pnp.array(
            pnp.random.uniform(low=0, high=pnp.pi, size=self.params_shape),
            requires_grad=True,
        )

        # self._params = pnp.random.uniform(
        # low=-pnp.pi, high=pnp.pi, size=self.params_shape
        # )

        return self._params

    @params.setter
    def params(self, params: pnp.ndarray):
        """Set the parameters for the circuit evaluation

        Args:
            params (pnp.ndarray): The parameters for the circuit evaluation.
        """
        self._params = params

    def __call__(self, params: pnp.ndarray) -> Operation:
        """Evaluate the circuit with given parameters.

        Args:
            params (pnp.ndarray): The parameters to be used for evaluating the circuit.

        Returns:
            Operation: The evaluation of the circuit.
        """
        return self._circuit_ansatz(params)

    @abstractmethod
    def _circuit_ansatz(self, params) -> Operation:
        raise NotImplementedError
