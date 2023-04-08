# %%
import pennylane as qml

from qflow.templates.circuits import QAOACircuit
from qflow.templates.examples.maxcut_circuit import maxcut_qaoa_example
from qflow.utils.maxcut_utils import get_maxcut_graph

# %%
num_nodes = 6
seed = 0

graph = get_maxcut_graph(num_nodes, seed=seed)
H, _ = qml.qaoa.maxcut(graph)
num_qubits = len(H.wires)

# circuit, H, min_energy = maxcut_qaoa_example(num_layers=2)

# %% [markdown]
# ## Vanilla QAOA

from qflow.qaoa.mixer_h import x_mixer

# %%
from qflow.templates.circuits import QAOACircuit
from qflow.templates.state_preparation import Plus

circuit = QAOACircuit(
    H, initial_state=Plus(num_qubits), mixer_h=x_mixer(num_qubits), num_layers=2
)
params = circuit.init()

# %%
print(qml.draw(circuit)(params))

# %% [markdown]
# ## XY-mixer QAOA

# %%
from qflow.qaoa.mixer_h import circular_xy_mixer
from qflow.templates.state_preparation import DickeState

circuit = QAOACircuit(
    H,
    initial_state=DickeState(num_qubits, 2),
    mixer_h=circular_xy_mixer(num_qubits),
    num_layers=2,
)

# %%
