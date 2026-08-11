# function to load our model parameters and the structural connectivity of the dataset

def load(PATH)
    metrics = ["ADC", "gFA", "number", "density"]
    J_i = {}
    wFFI = {}
    wLRE = {}
    SC_all = {}
    for metric in metrics:
        J_i[metric] = {}
        wFFI[metric] = {}   
        wLRE[metric] = {}
        SC_all[metric] = {}

        SC_all[metric]=np.load(f"{PATH}/SC_matrices/{metric}/{metric}_allsubj_Hagmann.npy", allow_pickle=True).item()
        for subj in SC_all[metric].keys():
            path = f"{PATH}/E_I tuning model/{metric}/{metric}_param_83_{subj}.npy"
            file = np.load(path, allow_pickle=True).item()
            J_i[metric][subj] = file["J_i"]
            wFFI[metric][subj] = file["wFFI"]
            wLRE[metric][subj] = file["wLRE"]

    region = np.load(f"{PATH}/reg_labels_Hagmann83.npy", allow_pickle=True)
    region_labels = np.load(f"{PATH}/Data/reg_labels_Hagmann129.npy", allow_pickle=True)

    return J_i, wFFI, wLRE, SC_all, region, region_labels

def visualize_BOLD(data)
    data = bold_signal.data.squeeze()
    print(len(data))

    time_s = bold_signal.time / 1000.0

    plt.figure(figsize=(12, 6))
    plt.plot(time_s, data, alpha=0.5)
    plt.title('Simulated BOLD Signal')
    plt.xlabel('Time Points')
    plt.ylabel('BOLD Signal')
    plt.legend()
    plt.grid()
    plt.show()