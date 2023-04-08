from abc import ABC, abstractmethod


class InitialState(ABC):
    """
    Abstract class for an initial state.
    """

    @abstractmethod
    def __init__(self, num_qubits, **kwargs):
        self.num_qubits = num_qubits
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self):
        pass

    def __repr__(self):
        return (
            f"Initial state: {self.__class__.__name__}, num_qubits: {self.num_qubits}"
        )
