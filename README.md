# UComFy

UComFy means **Uncertainty-aware Community and Feature Similarity Guided Rewiring**.

It fuses two ideas:

1. **ComFy rewiring**: convert a PyG graph to an undirected NetworkX graph, detect Louvain communities, allocate add/delete budgets to community pairs in proportion to `|C_i| * |C_j|`, add the highest cosine-similarity non-edges, and delete the lowest cosine-similarity existing edges.
2. **UnGSL uncertainty gating**: after rewiring, each undirected structural edge is used as directed message passing edges. For a PyG edge `source -> target`, UComFy computes

```text
w_{target<-source} = w_comfy(source,target) * psi(confidence[source] - threshold[target])
confidence[source] = exp(-entropy[source])
entropy = -sum softmax(logits) * log(softmax(logits) + eps)
raw = sigmoid(confidence[source] - threshold[target]) / 0.5
gate = raw if raw >= 1 else beta
edge_weight = w_comfy * (1 + gate_residual_alpha * (gate - 1))
```

The node-wise `threshold` vector is an `nn.Parameter` trained with the final classifier. The default threshold is `0.1`, which is a practical scale for `exp(-entropy)` confidence on multi-class citation datasets. At the default `gate_residual_alpha=1.0`, the gate forward values still match the UnGSL rule exactly; UComFy uses a straight-through gradient on the low-confidence beta branch so thresholds do not freeze when an initial threshold is conservative.
`gate_residual_alpha` defaults to `1.0`, which exactly preserves the original full-strength gate; smaller values are exploratory damped-gate ablations that keep edge weights closer to the rewired graph.

## Relation To ComFy

The original ComFy code mixes `from comsim import *` and `from onlysim import *`, where both modules define `modify_graph`. UComFy removes that ambiguity:

- `rewiring/comfy_rewire.py`: community + feature-similarity rewiring.
- `rewiring/feast_rewire.py`: pure feature-similarity rewiring used as a FeaSt-style ablation.

No original `ComFy-main` files are modified.

## Relation To UnGSL

The original UnGSL code hard-codes entropy files and devices such as `/home/hs/OpenGSL/...`, `cuda:0`, and `cuda:7`. UComFy keeps only the mechanism:

- `modules/ucomfy_gate.py`: clean PyG edge gate using `confidence[source] - thresholds[target]`.
- `modules/uncertainty.py`: entropy and confidence computed from a local pretrained GCN.
- `models.py`: `UComFyGCN` owns and trains the threshold parameters.

## Environment

Install PyTorch and PyTorch Geometric for your CUDA/CPU setup, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`nx_cugraph` is optional. If it is not installed, UComFy automatically falls back to NetworkX Louvain.

## Main Entry

```bash
python main.py --help
```

## Cora Example

```bash
python main.py --dataset Cora --rewire_method comfy --model UComFyGCN --budget_edges_add 100 --budget_edges_delete 100 --epochs 100 --pretrain_epochs 100 --device cuda
```

If CUDA is unavailable, use:

```bash
python main.py --dataset Cora --rewire_method comfy --model UComFyGCN --budget_edges_add 100 --budget_edges_delete 100 --epochs 100 --pretrain_epochs 100 --device cpu
```

## Citeseer Example

```bash
python main.py --dataset Citeseer --rewire_method comfy --model UComFyGCN --budget_edges_add 100 --budget_edges_delete 100 --epochs 100 --pretrain_epochs 100 --device cuda
```

## Pubmed Example

```bash
python main.py --dataset Pubmed --rewire_method comfy --model UComFyGCN --budget_edges_add 300 --budget_edges_delete 300 --epochs 100 --pretrain_epochs 100 --device cuda
```

## Heterophily NPZ Data

Supported npz names include:

- `cornell.npz`
- `texas.npz`
- `wisconsin.npz`
- `chameleon_filtered.npz`
- `squirrel_filtered.npz`
- `actor.npz`

Pass the directory containing those files:

```bash
python main.py --dataset cornell.npz --hetero_data_path D:\path\to\heterophilous-graphs\data --rewire_method comfy --model UComFyGCN --device cpu
```

If the file is missing, the loader raises a clear error pointing back to `--hetero_data_path`.

## Ablations

Original graph + plain GCN:

```bash
python main.py --dataset Cora --rewire_method none --model GCN --device cuda
```

ComFy + plain GCN:

```bash
python main.py --dataset Cora --rewire_method comfy --model GCN --budget_edges_add 100 --budget_edges_delete 100 --device cuda
```

ComFy + UComFyGCN:

```bash
python main.py --dataset Cora --rewire_method comfy --model UComFyGCN --budget_edges_add 100 --budget_edges_delete 100 --device cuda
```

FeaSt + UComFyGCN:

```bash
python main.py --dataset Cora --rewire_method feast --model UComFyGCN --budget_edges_add 100 --budget_edges_delete 100 --device cuda
```

## Files

- `main.py`: full experiment pipeline, split loop, CSV logging.
- `arguments.py`: command line arguments.
- `dataloader.py`: Planetoid/Amazon/Coauthor and npz loading.
- `rewiring/comfy_rewire.py`: ComFy community-budget rewiring.
- `rewiring/feast_rewire.py`: pure feature-similarity rewiring.
- `modules/ucomfy_gate.py`: UnGSL-style directional edge gate.
- `modules/uncertainty.py`: entropy, confidence, and cache helpers.
- `models.py`: baseline `GCN` and `UComFyGCN`.
- `train.py`: pretraining and final training loops.
- `utils/metrics.py`: homophily, adjusted homophily, and Louvain NMI.

Results are appended to `results/ucomfy_results.csv` by default. Entropy/confidence caches are stored under `results/cache` unless `--no_cache_entropy` is used.

## Diagnostics

Each run prints light fusion diagnostics:

- rewiring edge counts before/after plus added/deleted edges,
- entropy and confidence min/max/mean and finite checks,
- UComFy edge weight min/max/mean and finite checks,
- whether thresholds are `nn.Parameter`, require gradients, are present in the optimizer, and changed or received gradients after training,
- whether `UComFyGCN` passed `edge_weight` into `GCNConv`.

## Large Graph Note

For small graphs, non-edges inside each community pair are streamed exactly. For large community pairs, the code avoids materializing all non-edges and falls back to row-wise top-k candidate search. This keeps memory bounded but can miss some global top-similarity non-edges; the CSV and console warnings make that degradation visible.
