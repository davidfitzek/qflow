import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from pennylane.operation import Operation

from qflow.qaoa.mixer_h import *
from qflow.templates.circuits import AbstractCircuit
from qflow.templates.state_preparation import *


class QAOACircuit(AbstractCircuit):
    """This class implements the Quantum Approximate Optimization Algorithm (QAOA) circuit.

    This circuit consists of a series of alternating blocks of time evolution
    with two Hamiltonians, the cost Hamiltonian and the mixer Hamiltonian, followed
    by a final measurement in the computational basis.

    Args:
        H (qml.Hamiltonian): The cost Hamiltonian of the optimization problem.
        initial_state (qml.qnode, optional): The initial state of the circuit. Default is the Plus state.
        mixer_h (qml.qnode, optional): The mixer Hamiltonian. Default is the x_mixer.
        num_layers (int, optional): The number of blocks of time evolution. Default is 1.

    Attributes:
        H (qml.Hamiltonian): The cost Hamiltonian of the optimization problem.
        initial_state (qml.qnode): The initial state of the circuit.
        mixer_h (qml.qnode): The mixer Hamiltonian of the optimization problem.
        num_qubits (int): The number of qubits in the Hamiltonian.
        num_layers (int): The number of blocks of time evolution.

    """

    def __init__(
        self,
        H: qml.Hamiltonian,
        initial_state: qml.qnode = Plus,
        mixer_h: qml.qnode = x_mixer,
        num_layers: int = 1,
    ):
        self.H = H
        self.initial_state = initial_state
        self.mixer_h = mixer_h
        self.num_layers = num_layers
        self.num_qubits = len(self.H.wires)

        assert len(self.mixer_h.wires) == len(
            self.H.wires
        ), f"N qubits mixer_h: {len(self.mixer_h.wires)} N qubits H: {len(self.H.wires)} "

    @property
    def wires(self):
        return self.H.wires

    def init(self, seed=None):
        """Return the initial parameters of the circuit.

        Args:
            rng_key (None or np.random.Generator, optional): The random number generator seed. Default is None.

        Returns:
            numpy.ndarray: The initial parameters of the circuit.
        """
        if seed is not None:
            pnp.random.seed(seed)

        return self._params(seed)

    def _params(self, seed=None):
        gamma_init = pnp.array(2 * np.pi * np.random.rand(self.num_layers))
        beta_init = pnp.array(np.pi * np.random.rand(self.num_layers))
        x0 = pnp.concatenate((gamma_init, beta_init), axis=0)
        return x0

    def __call__(self, params: pnp.ndarray) -> Operation:
        """Evaluates the circuit at the given parameters.

        Args:
            params (numpy.ndarray): The parameters of the circuit.

        Returns:
            Operation: The evaluation of the circuit.
        """
        return self._circuit_ansatz(params)

    def _circuit_ansatz(self, params) -> Operation:
        """Perform the actual circuit evaluation with given parameters.

        Args:
            params (np.ndarray): The parameters to be used for evaluating the circuit.

        Returns:
            Operation: The evaluation of the circuit.

        Raises:
            AssertionError: If `params` does not have the expected length, which
            is len(parmams) != 2 * self.num_layers.
        """
        # assert isinstance(
        # params, pnp.tensor
        # ), f"Params must be of type pnp.tensor, but got {type(params)}"
        assert 2 * self.num_layers == len(
            params
        ), f"{len(params)} != {2 * self.num_layers}"

        self.initial_state()
        beta_list = params[self.num_layers :]
        gamma_list = params[: self.num_layers]
        for beta, gamma in zip(beta_list, gamma_list):
            qml.ApproxTimeEvolution(self.H, gamma, 1)
            qml.ApproxTimeEvolution(self.mixer_h, beta, 1)
