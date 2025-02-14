# raysearch_case

## Overview

This repo contains

- A script `experiments.py` to train and evaluate models from MNIST test data.
- A tracking module `experiment_tracker` integrated with the script to log information to a database `experiment_db` and write model checkpoints to local file storage.
- A FastAPI web app `main.py` to visualize the information in the database.

The anatomy of the repo laid out below:

```txt
.
├── README.md            │    This file
├── requirements.txt     │    Python package requirements
│                        │
├── experiments.py       │    Original script
├── root                 │    Created / managed by original script
│   ├── MNIST            │     - Raw testdata
│   │   └── raw          │
│   │       └── ...      │
│   ├── *.pt             │     - Tensors for labeled test data
│   └── *.pth            │     - Working models
│                        │
│                        │
├── experiment_tracker   │   Tracking module
│   ├── __init__.py      │
│   ├── database.py      │    
│   ├── inspect.py       │
│   └── tracker.py       │
│                        │
├── artifacts            │   Created / managed by tracking module
│   ├── 1                │     - Experiment ID dir
│   │   ├── epoch_1.pth  │         - Checkpoints
│   │   ├── epoch_2.pt   │
│   │   └── ...          │
│   └── ...              │
│                        │
├── experiments.db       │   Created / managed by tracking module
└── main.py              │   FastAPI web app

```

## Run instructions

### Set-up

Built using Python 3.12.8

```bash
pip install -r requirements.txt
```

### Run

```bash
python experiments.py
```

### Launch web app

```bash
uvicorn main:app --reload
```

Web app should be accessible at `http://localhost:8000/experiments`. The experiment IDs of the table are hyperlinks.
