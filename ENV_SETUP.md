# UComFy Environment Setup Notes

## Current DL1 Test Status

The user later allowed testing and repair with:

```text
D:\anaconda2022_10\envs\DL1\python.exe
```

No package install, uninstall, upgrade, or downgrade was performed in DL1. The existing DL1 environment was sufficient to run the project on CPU after setting this process-local variable for each Python command:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

This variable is needed because direct `import torch` triggers an OpenMP duplicate-runtime error in the current DL1 process. It does not modify the environment permanently.

Observed runtime checks:

| Check | Result |
| --- | --- |
| Python path | `D:\anaconda2022_10\envs\DL1\python.exe` |
| `import torch; torch.__version__` | `2.5.1` |
| `torch.cuda.is_available()` | `True` |
| `torch_geometric.__version__` | `2.6.1` |
| Smoke device used | `cpu` |

Package metadata from `pip list` / `pip freeze` still reports `torch==2.2.1+cu121` and PyG extension wheels compiled for the `pt21cu121` stack. At runtime, PyG warns that `torch-scatter`, `torch-sparse`, `torch-cluster`, and `torch-spline-conv` cannot be loaded and disables them. The tested CPU Cora flows still pass with PyG's available fallback path.

Exported records:

- `environment_DL1_after.yml`
- `pip_freeze_DL1_after.txt`

## Original codex-gnn Environment Setup Plan

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

## Step 3: Actual Dependency Check Results

Checked with `D:\anaconda2022_10\envs\codex-gnn\python.exe`:

| Check | Result |
| --- | --- |
| `import torch` | Missing |
| `import torch_geometric` | Missing |
| `import networkx, sklearn, numpy` | Missing at `networkx` |
| Python | `3.10.20` |
| GPU | NVIDIA RTX 3060, driver reports CUDA 13.1 |

The CUDA driver is new enough to run CUDA 12.1 wheels, so the first planned install uses the requested PyTorch 2.2.1 + cu121 stack.

## Step 4: Planned Install Commands

All commands target only `codex-gnn` by invoking its explicit interpreter.

| Package | Version | Command | Reason |
| --- | --- | --- | --- |
| `torch`, `torchvision`, `torchaudio` | `2.2.1+cu121`, `0.17.1+cu121`, `2.2.1+cu121` | `D:\anaconda2022_10\envs\codex-gnn\python.exe -m pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121` | Required by all models and training loops; CUDA wheel matches available NVIDIA GPU/driver |
| PyG extension wheels | Matching `torch-2.2.1+cu121` index | `D:\anaconda2022_10\envs\codex-gnn\python.exe -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cu121.html` | Optional compiled PyG acceleration/ops used by PyG components |
| `torch_geometric`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `networkx`, `tqdm`, `matplotlib` | `torch_geometric==2.5.2`, `numpy==1.26.4`, other packages resolver-selected | `D:\anaconda2022_10\envs\codex-gnn\python.exe -m pip install torch_geometric==2.5.2 numpy==1.26.4 scipy scikit-learn pandas networkx tqdm matplotlib` | Required for PyG graph data/model code, metrics, CSV/logging helpers, and plotting-compatible dependencies |

If the CUDA wheel install fails, stop and record the failure before trying the CPU fallback commands from the user instructions. Do not install DGL or OpenGSL; UComFy does not depend on them.

## Step 5: Install Attempt Status

Attempted command:

```bat
D:\anaconda2022_10\envs\codex-gnn\python.exe -m pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Result:

- The command required sandbox escalation because it writes into `D:\anaconda2022_10\envs\codex-gnn`.
- The escalation approval did not return before the tool deadline on two attempts.
- No packages were installed by Codex after this failure.
- Current `pip freeze` shows only `packaging==26.0`.

Next manual step:

- Run the planned install commands from Step 4 in a terminal where writes to `codex-gnn` are allowed.

## DL1 Test Environment Notes

The user later allowed testing with:

```text
D:\anaconda2022_10\envs\DL1\python.exe
```

Observed DL1 packages relevant to UComFy:

| Package | Version |
| --- | --- |
| `torch` | `2.2.1+cu121` |
| `torchvision` | `0.17.1+cu121` |
| `torchaudio` | `2.2.1+cu121` |
| `torch-geometric` | `2.6.1` |
| `torch-scatter` | `2.1.2+pt21cu121` |
| `torch-sparse` | `0.6.18+pt21cu121` |
| `torch-cluster` | `1.6.3+pt21cu121` |
| `torch-spline-conv` | `1.2.2+pt21cu121` |
| `numpy` | `1.26.4` |
| `scipy` | `1.13.1` |
| `scikit-learn` | `1.6.1` |
| `pandas` | `2.2.3` |
| `networkx` | `3.2.1` |
| `tqdm` | `4.67.1` |
| `matplotlib` | `3.5.1` |

No package install or uninstall has been performed in DL1 so far.

Direct `import torch` in DL1 currently fails with:

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

For smoke testing only, commands set the process-local environment variable:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

This is not a package installation or environment mutation; it only affects the current command process.

## Code Fix From Smoke Diagnostics

The first `ComFy + UComFyGCN` smoke run with `init_threshold=0.5` produced valid confidence values around `0.143`, but every edge fell into the beta branch:

```text
edge_weight min=max=mean=0.2
threshold grad norm=0.0
threshold max abs delta=0.0
```

This meant the fusion executed but the uncertainty gate was effectively frozen for Cora.

Fix applied in project code:

- `arguments.py`: changed default `--init_threshold` from `0.5` to `0.1`, matching the scale of `exp(-entropy)` confidence for multi-class datasets.
- `modules/ucomfy_gate.py`: preserved the exact UnGSL forward rule `raw if raw >= 1 else beta`, but added a straight-through gradient for the low-confidence beta branch so thresholds can still receive learning signal.
