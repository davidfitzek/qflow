from abc import ABC, abstractmethod

import numpy as np
from pennylane.operation import Operation


class AbstractCircuit(ABC):
    """Abstract base class for quantum circuits."""

    def __init__(self):
        """Initialize the circuit.

        Note:
            This is an abstract method that should be overridden by subclasses.
        """
        pass

    def __call__(self) -> Operation:
        """Return the circuit as a PennyLane operation.

        Returns:
            Operation: The PennyLane operation representing the circuit.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def __repr__(self):
        """Return a string representation of the circuit.

        Returns:
            str: The string representation of the circuit.
        """
        return f"{self.__class__.__name__}"

    @property
    def wires(self):
        """Return the wires of the circuit.

        Returns:
            Union[int, Tuple[int]]: The wires of the circuit.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("The circuit does not have wires.")

    @wires.setter
    def wires(self, wires):
        self._wires = wires

    def init(self, seed=None) -> np.ndarray:
        """Initialize the parameters of the circuit.

        Args:
            seed (int, optional): The random seed used to initialize the parameters.

        Returns:
            np.ndarray: The initialized parameters of the circuit.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def _circuit_ansatz(self, params) -> Operation:
        """Return the circuit ansatz of the circuit.

        Args:
            params (np.ndarray): The parameters of the circuit.

        Returns:
            Operation: The circuit ansatz of the circuit.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("The circuit does not have a circuit ansatz.")

    # def update_circuit_ansatz(
    #     self, circuit_ansatz: Operation, wires: List
    # ) -> Operation:
    #     """Update the circuit ansatz of the circuit.

    #     Args:
    #         circuit_ansatz (Operation): The updated circuit ansatz.
    #         num_qubits (int): The number of qubits in the circuit.

    #     Returns:
    #         Operation: The updated circuit ansatz of the circuit.
    #     """
    #     # check if wires is instance of List or Tuple
    #     assert not isinstance(
    #         wires, int
    #     ), f"wires must be a list of integers but got{type(wires)}"

    #     self._wires = wires
    #     self._circuit_ansatz = circuit_ansatz
    #     self.num_qubits = len(self._wires)
