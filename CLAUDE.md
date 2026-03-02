# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Federated Learning (FL) Unlearning** research project. It studies how to remove the influence of a specific poisoned client (backdoor attacker) from a trained federated learning model, comparing multiple unlearning strategies.

The poisoned client is always **client 0**, which has `poisoned_percent` (default 90%) of its data backdoored. Clients 1–4 hold clean data.

## Running Experiments

### Single case
```bash
python case0.py                          # Baseline FL training
python case1.py                          # Retrain from scratch (no poisoned client)
python case2.py                          # Continue training excluding poisoned client
python case3.py                          # PGA (Projected Gradient Ascent)
python case4.py                          # FedEraser
python case5.py                          # Label Flipping
```

### All cases with custom parameters
```bash
python run_all_cases.py                              # Run cases 0–5 with defaults
python run_all_cases.py --cases 0-2,4               # Selective cases
python run_all_cases.py --num_rounds 20 --lr 0.01   # Custom parameters
```

### Key configuration parameters (passed as CLI args)
- `--dataset [mnist|cifar10|cifar100]` (default: cifar10)
- `--num_rounds` — initial FL training rounds (default: 20)
- `--num_unlearn_rounds` — unlearning phase rounds (default: 5)
- `--num_post_training_rounds` — post-unlearning retraining (default: 20)
- `--poisoned_percent` — fraction of client 0's data poisoned (default: 0.9)
- `--is_onboarding True|False` — whether to include an onboarding phase
- `--saved True|False` — whether to save model weights per round

### Results analysis
Open `result_sample/usage.ipynb` in Jupyter to visualize metrics from saved `.pkl` files.

## Code Quality

Pre-commit hooks enforce formatting on every commit:
```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files   # Run manually
```

Tools: **Black** (formatter), **isort** (import sorting), **Flake8** (linter, max line length 88).

## Architecture

### Experiment flow

```
case0.py → Standard FL training (all 5 clients, FedAvg, save models)
    ↓
case1–5.py → Load case0 state, apply unlearning method, evaluate
```

Cases 1–5 all depend on case0's saved models in `results/models/case0/`. Case 0 must be run first.

### Module roles

| Path | Purpose |
|------|---------|
| `config.py` | Argparse-based config; auto-generates output filenames |
| `utils/dataloader.py` | Loads MNIST/CIFAR-10/CIFAR-100; injects backdoor via ART library into client 0's data; creates clean + poisoned test sets |
| `utils/model.py` | Model factory (`get_model()`): FLNet (MNIST), CNNCifar (CIFAR-10/100), ResNet18, Cifar100 (ResNet50) |
| `utils/clients.py` | `client_train()`: local SGD; supports `is_flip` flag for label-flipping attack |
| `utils/server.py` | `FedAvg()` aggregation; `test()` evaluates clean and backdoor accuracy |
| `utils/utils.py` | `get_results()`, `save_param()`, `update_results()`, `load_results()`; Utils class for model distance metrics |
| `utils/meter.py` | `Meter` (loss/acc tracking), `EvaluationMetrics` (L2, cosine sim, SAPE, FC similarity) |
| `unlearn/federaser.py` | FedEraser algorithm using stored client update history |
| `unlearn/pga.py` | PGA: `compute_ref_vec()` + `unlearn()` with gradient ascent and distance constraints |
| `unlearn/flipping.py` | Wrapper: client 0 trains with random labels |

### Output structure

```
results/
├── case{X}_*.pkl          # Metrics: train/val loss and accuracy (clean + backdoor)
└── models/case{X}/
    ├── case{X}_*_round{r}.pt          # Global model per round
    └── client{i}/case{X}_*_round{r}.pt   # Per-client models per round
```

### Evaluation metrics

Each `.pkl` contains `train` and `val` dicts with:
- `loss`/`acc` → keys: `"avg"`, `"clean"`, `"backdoor"`, `0`, `1`, …, `4`

Success criterion: **high clean accuracy, low backdoor accuracy** after unlearning.

### Reproducibility

Seeds are fixed: `numpy.random.seed(42)`, `torch.manual_seed(42)`, `torch.backends.cudnn.deterministic = True`.
