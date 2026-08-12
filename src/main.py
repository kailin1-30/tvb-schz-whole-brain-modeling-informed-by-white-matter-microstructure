import os

import numpy as np
import jax

from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun, BoundedSolver
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.observations.tvb_monitors.bold import Bold

from network import ReducedWongWangEIB, EIBLinearCoupling
from tools import load

PATH = os.path.join(os.path.dirname(__file__), "..", "data")
PATH_result = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(PATH_result, exist_ok=True)

J_i, wFFI, wLRE, SC_all, region, region_labels = load(PATH)

weights = SC_all['ADC']['schz1'][83]

graph    = DenseGraph(weights, region_labels=region_labels)
dynamics = ReducedWongWangEIB(J_i=J_i['ADC']['schz1'])
coupling = EIBLinearCoupling(incoming_states=["S_e"])
coupling.params.wLRE = wLRE['ADC']['schz1']
coupling.params.wFFI = wFFI['ADC']['schz1']

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

state_long.dynamics.J_i            = J_i['ADC']['schz1']
state_long.coupling.coupling.wLRE  = wLRE['ADC']['schz1']
state_long.coupling.coupling.wFFI  = wFFI['ADC']['schz1']

print("Running forward simulation...")

raw_result = model_long(state_long)
bold_signal = bold_monitor(raw_result)

data = {'data': bold_signal.data.squeeze(), 'time': bold_signal.time}
np.save(f"{PATH_result}/BOLD_signal.npy", data)

