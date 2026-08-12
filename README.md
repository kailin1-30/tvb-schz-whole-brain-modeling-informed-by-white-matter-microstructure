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

The notebook/script currently reference local absolute paths (e.g. structural connectivity matrices, region labels, parameter files) that live outside this repository. Update those paths to point to your own copies of the data before running.

