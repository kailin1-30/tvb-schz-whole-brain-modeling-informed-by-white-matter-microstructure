import numpy as np
import jax
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun, BoundedSolver
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.external_input import PulseInput
from tvboptim.observations.tvb_monitors.bold import Bold

from network import ReducedWongWangEIB, EIBLinearCoupling
from tools import load

PATH = "/Users/lin/Documents/Bachelor thesis /"
J_i, wFFI, wLRE, SC_all, region, region_labels = load(PATH)

# define labels for classification of participants
labels = []
subject_ids = []
results = {}
# Transient time:
trans = 5*60_000
metric = 'ADC'
SUBJECTS = SC_all[metric].keys()
AMPLITUDE = 0.01
SPLITS = 10
DURATION = 500 #in ms
SEED = 100
PATH = f"/Users/lin/Documents/Bachelor thesis /hallucinations/simulations/results/S_i_S_e/10subj_{DURATION}_{AMPLITUDE}_{SEED}.npy"
PATH_LABEL = f"/Users/lin/Documents/Bachelor thesis /hallucinations/simulations/results/S_i_S_e/10subj_{DURATION}_{AMPLITUDE}_{SEED}_labels.npy"


subjects = SUBJECTS
metrics = ['ADC']
for metric in metrics:
        results[metric] = {}
        for subj in subjects:
            print(subj)
            # print(type(wFFI[metric][subj]), type(wLRE[metric][subj]), type(J_i[metric][subj]))
            weights = SC_all[metric][subj][83]
            graph    = DenseGraph(weights, region_labels=region_labels)
            dynamics = ReducedWongWangEIB(J_i=jnp.array((J_i[metric][subj])))
            coupling = EIBLinearCoupling(incoming_states=["S_e"])
            coupling.params.wLRE = jnp.array(wLRE[metric][subj])
            coupling.params.wFFI = jnp.array(wFFI[metric][subj])
            results[metric][subj] = {}


            # define the stimulus:
            stim_dur = DURATION

            # define noise and set seed:
            # first 50 with stimulus and last 50 without stimulus, to see the difference in BOLD signal
            # 10 splits
            for seed in range(SEED):
                results[metric][subj][seed] = {}
                # for ampl in [0.0, 10]:
                noise = AdditiveNoise(sigma=0.01, apply_to="S_e", key=jax.random.key(seed))

                network = Network(
                            dynamics=dynamics,
                            coupling={"coupling": coupling},
                            graph=graph,
                            noise=noise
                        )
                # Transient simulation to reach steady state
                dt     = 4.0
                solver = BoundedSolver(Heun(), low=0.0, high=1.0)

                model_init, state_init = prepare(network, solver, t1=5*60_000, dt=dt)
                print("Running transient for seed " + str(seed) + "..." )
                result_init = jax.block_until_ready(model_init(state_init))
                network.update_history(result_init)

                if seed > 49:
                        ampl = AMPLITUDE
                        group = 1
                else:
                        ampl = 0.0
                        group = 0
                print(seed)

                dur = stim_dur
                labels.append(group)
                subject_ids.append(subj)

                bold_TR = 720.0
                t1_long = 5000

                stim_region_idx = [27,31, 71, 68, 72, 30]
                stim_ampl_l = np.zeros(len(region_labels))
                stim_ampl_l[stim_region_idx] = ampl
                stim = PulseInput(onset=t1_long/2, duration=dur, amplitude=stim_ampl_l)
                results[metric][subj][seed][(ampl, dur)] = []

                network_stim = Network(
                        dynamics=dynamics,
                        coupling={"coupling": coupling},
                        graph=graph,
                        noise=noise,
                        external_input={"stim":  stim}
                    )
                network_stim.update_history(result_init)


                bold_monitor = Bold(
                    period=bold_TR,
                    downsample_period=dt,
                    voi=0,
                    history=result_init
                )

                model_long, state_long = prepare(network_stim, solver, t1=t1_long, dt=dt, )


                state_long.dynamics.J_i            = J_i[metric][subj]
                state_long.coupling.coupling.wLRE  = wLRE[metric][subj]
                state_long.coupling.coupling.wFFI  = wFFI[metric][subj]

                print(f"Running forward simulation, {seed}, {subj}, {ampl} ...")

                raw_result = model_long(state_long)

                bold_signal = bold_monitor(raw_result)
                # data = bold_signal.data.squeeze()
                data = raw_result.data.squeeze()
                # data_diff = jnp.array(data - temp)

                # visualization
                time_s = bold_signal.time / 1000.0
                results[metric][subj][seed][(ampl, dur)].append(data)
                print(results[metric][subj][seed].keys())

                # plt.figure(figsize=(12, 6))
                # for idx in stim_region_idx:
                #     plt.plot(time_s, data_diff[:, idx], alpha=0.5)
                #     # plt.plot(time_s, data[:, idx], label=region_labels[idx], linestyle='--')
                #     # plt.plot(time_s, temp[:, idx], label=region_labels[idx], alpha=0.3)
                # plt.title('Simulated BOLD Signal, stim_dur: ' + str(dur) + ' Amplitude: ' + str(ampl) + "seed" + str(seed))
                # plt.xlabel('Time Points')
                # plt.ylabel('BOLD Signal')
                # plt.legend()
                # plt.grid()
                # # plt.savefig(f"/Users/lin/Documents/Bachelor thesis /hallucinations/simulations/results/BOLD/diff/{metric}_{subj}_seed{seed}_stimdur{dur}_ampl{ampl}.pdf")
                # plt.show()


                ## plot the differences

                # plt.figure(figsize=(12, 6))
                # for idx in stim_region_idx:
                #     plt.plot(time_s, data[:, idx], label=region_labels[idx])
                # plt.title('Simulated BOLD Signal, Duration: ' + str(dur) + 'ms, Amplitude: ' + str(ampl) + ", seed" + str(seed))
                # plt.xlabel('Time Points')
                # plt.ylabel('BOLD Signal')
                # plt.legend()
                # plt.grid()
                # plt.savefig(f"/Users/lin/Documents/Bachelor thesis/hallucinations/simulations/results/BOLD/{metric}_{subj}_seed{seed}_stimdur{dur}_ampl{ampl}.pdf")
                # plt.show()

                # temp = data

labels      = np.array(labels)       # shape: [number of subjects * number of stim conditions * number of seeds]
subject_ids = np.array(subject_ids)
np.save(f"{PATH}.npy", results)
np.save(f"{PATH_LABEL}.npy", labels)
