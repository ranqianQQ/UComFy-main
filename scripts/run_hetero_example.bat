@echo off
cd /d %~dp0\..
python main.py --dataset cornell.npz --hetero_data_path D:\path\to\heterophilous-graphs\data --rewire_method comfy --model UComFyGCN --budget_edges_add 50 --budget_edges_delete 50 --epochs 100 --pretrain_epochs 100 --device cpu
