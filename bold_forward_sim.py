# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import jax
import jax.numpy as jnp
import copy
import optax
from scipy import io
import equinox as eqx

# Import from tvboptim
from tvboptim.types import Parameter, BoundedParameter
from tvboptim.types.stateutils import show_parameters
from tvboptim.utils import set_cache_path, cache
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingLossCallback

# Network dynamics imports
from tvboptim.experimental.network_dynamics import Network, solve, prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling, FastLinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun, BoundedSolver
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.data import load_structural_connectivity, load_functional_connectivity

# BOLD monitoring
from tvboptim.observations.tvb_monitors.bold import Bold

# Observation functions
from tvboptim.observations.observation import compute_fc, fc_corr, rmse

# Caching utilities
from tvboptim.utils import set_cache_path, cache

# load the parameters and the structural connectivity
metrics = ["ADC", "gFA", "number", "density"]

path = "/Users/lin/Documents/Bachelor thesis /E_I tuning model/ADC/ADC_param_83_schz1.npy"
file = np.load(path, allow_pickle=True).item()
J_i = jnp.array(file["J_i"])
wFFI = jnp.array(file["wFFI"])
wLRE = jnp.array(file["wLRE"])
file_SC = np.load("/Users/lin/Documents/Bachelor thesis /SC_matrices/ADC/ADC_allsubj_Hagmann.npy", allow_pickle=True).item()
print(file_SC["schz1"].keys())
weights = jnp.array(file_SC["schz1"][83])
region_labels = np.load("/Users/lin/Documents/Bachelor thesis /Data/reg_labels_Hagmann83.npy", allow_pickle=True)

print(len(weights), len(J_i), len(wFFI), len(wLRE)) 
# prepare network and run forward simulation

graph    = DenseGraph(weights, region_labels=region_labels)
dynamics = ReducedWongWangEIB(J_i=J_i)
coupling = EIBLinearCoupling(incoming_states=["S_e"])
coupling.params.wLRE = wLRE
coupling.params.wFFI = wFFI

network = Network(
    dynamics=dynamics,
    coupling={"coupling": coupling},
    graph=graph,
    noise=AdditiveNoise(sigma=0.01, apply_to="S_e")
)

# Transient simulation to reach steady state
dt     = 4.0
solver = BoundedSolver(Heun(), low=0.0, high=1.0)

model_init, state_init = prepare(network, solver, t1=5*60_000, dt=dt)
print("Running transient...")
result_init = jax.block_until_ready(model_init(state_init))
network.update_history(result_init)

bold_TR = 720.0
t1_long = 60_000  # 1 Minute

bold_monitor = Bold(
    period=bold_TR,
    downsample_period=dt,
    voi=0,
    history=result_init
)

model_long, state_long = prepare(network, solver, t1=t1_long, dt=dt)

state_long.dynamics.J_i            = J_i
state_long.coupling.coupling.wLRE  = wLRE
state_long.coupling.coupling.wFFI  = wFFI

print("Running forward simulation...")

raw_result = model_long(state_long)
bold_signal = bold_monitor(raw_result)
firing_rate = raw_result.dynamics.S_e.mean(axis=0)

# visualization

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# BOLD Zeitreihe
axes[0].plot(bold_signal)
axes[0].set_title("BOLD Signal (alle Regionen)")
axes[0].set_xlabel("Timepoints (TR = 720ms)")
axes[0].set_ylabel("BOLD")

# Firing Rate pro Region
axes[1].bar(range(n_nodes), firing_rate)
axes[1].set_title("Mittlere Firing Rate pro Region (S_e)")
axes[1].set_xlabel("Region")
axes[1].set_ylabel("S_e")

plt.tight_layout()
plt.savefig("bold_firing_rate.png", dpi=150)
plt.show()