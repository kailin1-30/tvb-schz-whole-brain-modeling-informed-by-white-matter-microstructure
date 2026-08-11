# TVB-whole-brain-modeling simulation of hallucination in schizophrenia

This repository contains the complimentary code for my preliminary work during my internship for a schizophrenia project simulating hallucinations. 
The input are the model parameter using the TVBoptim EI-Tuning Model with FIC. 
We are testing auditory and auditory verbal hallucinations with the respective regions being: 
temporalpole, superiortemporal, bankssts

## Repository structure

```
.
├── notebooks/
│   └── bold_forward_sim.ipynb   # Forward BOLD simulation, exploratory analysis
├── src/
│   └── bold_forward_sim.py      # Script version of the forward simulation
├── requirements.txt
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

`tvboptim` is an internal lab package and is not published on PyPI — install it from its own source before running the code here.

## Data

The notebook/script currently reference local absolute paths (e.g. structural connectivity matrices, region labels, parameter files) that live outside this repository. Update those paths to point to your own copies of the data before running.

