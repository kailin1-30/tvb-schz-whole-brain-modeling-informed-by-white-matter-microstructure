# TVB-whole-brain-modeling simulation of hallucination in schizophrenia

This repository contains the complimentary code for my preliminary work during my internship for a schizophrenia project simulating hallucinations. 
The input are the model parameter using the TVBoptim EI-Tuning Model with FIC. 
We are testing auditory and auditory verbal hallucinations with the respective regions being: 
temporalpole, superiortemporal, bankssts

## Repository structure

```
.
├── notebooks/
│   ├── bold_forward_sim.ipynb        # Original monolithic exploratory notebook
│   ├── visualization.ipynb           # Plots the BOLD signal saved by src/main.py
│   └── classification_analysis.ipynb # Accuracy + FP/FN stats and plots from src/classify.py output
├── src/
│   ├── network.py       # ReducedWongWangEIB dynamics + EIBLinearCoupling
│   ├── tools.py          # load() for model params/SC, visualize_BOLD()
│   ├── main.py            # Forward BOLD simulation (single subject) -> outputs/BOLD_signal.npy
│   ├── experiment.py      # Multi-subject/seed stimulation experiment -> results + labels
│   └── classify.py        # SGD classifier on experiment results -> accuracies, FP/FN
├── outputs/              # Local run outputs (gitignored)
├── data/                 # Local input data (gitignored, see Data section below)
├── requirements.txt
└── README.md
```

Run order for the classification pipeline: `main.py` (single-subject sanity check) → `experiment.py` (multi-subject/seed stimulation runs, saves `results`/`labels`) → `classify.py` (trains classifier on those results, saves accuracies + FP/FN) → `classification_analysis.ipynb` (stats + plots).

## Requirements

```bash
pip install -r requirements.txt
```

`tvboptim` is an internal lab package and is not published on PyPI — install it from its own source before running the code here.

## Data

All code reads input data from `data/` (relative to the repo root) and writes results to `outputs/` — neither is committed to git. `data/` is expected to contain:

```
data/
├── SC_matrices/
│   ├── ADC/ADC_allsubj_Hagmann.npy
│   ├── gFA/gFA_allsubj_Hagmann.npy
│   ├── number/number_allsubj_Hagmann.npy
│   └── density/density_allsubj_Hagmann.npy
├── E_I tuning model/
│   ├── ADC/ADC_param_83_<subject>.npy   # one file per subject
│   ├── gFA/...
│   ├── number/...
│   └── density/...
└── Data/
    └── reg_labels_Hagmann83.npy
```

This data is shared with other notebooks/analyses outside this repo, so rather than copying it in, `data/` holds symlinks to the original folders:

```bash
mkdir -p data
ln -s /path/to/SC_matrices          "data/SC_matrices"
ln -s "/path/to/E_I tuning model"   "data/E_I tuning model"
ln -s /path/to/Data                 "data/Data"
```

