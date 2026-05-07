# codex-gnn Environment Setup Plan

This project must use only:

```text
D:\anaconda2022_10\envs\codex-gnn\python.exe
```

No commands should use base, default `python`, DL1, or any other environment.

## Step 1: Create Environment If Missing

Package/runtime:

- `python=3.10`

Command:

```bat
D:\anaconda2022_10\Scripts\conda.exe create -n codex-gnn python=3.10 -y
```

Reason:

- `codex-gnn` is the dedicated environment requested for UComFy checks.

## Step 2: Check Existing Versions Before Installing Anything

Commands:

```bat
D:\anaconda2022_10\envs\codex-gnn\python.exe -c "import sys; print(sys.executable)"
D:\anaconda2022_10\envs\codex-gnn\python.exe -c "import torch; print(torch.__version__)"
D:\anaconda2022_10\envs\codex-gnn\python.exe -c "import torch_geometric; print(torch_geometric.__version__)"
D:\anaconda2022_10\envs\codex-gnn\python.exe -c "import networkx, sklearn, numpy; print('ok')"
```

## Step 3: Conditional Dependency Installation

No dependency installation is planned until Step 2 confirms what is missing.

If missing and compatible, install only into `codex-gnn` with fixed versions. Candidate packages and reasons:

| Package | Version | Command | Reason |
| --- | --- | --- | --- |
| `torch` | TBD after Python/CUDA check | TBD | Required by GCN/UComFyGate/training |
| `torch-geometric` | TBD after torch check | TBD | Required for PyG `Data`, `GCNConv`, datasets, graph utils |
| `networkx` | TBD after check | TBD | Required for Louvain and graph rewiring |
| `numpy` | TBD after check | TBD | Required for feature similarity and metrics |
| `scikit-learn` | TBD after check | TBD | Required for NMI |
| `scipy` | TBD after check | TBD | Common PyG/scikit dependency |
| `tqdm` | TBD after check | TBD | Optional progress dependency |

If PyTorch/PyG/CUDA compatibility is unclear or conflicting, stop and report the conflict before installing.
