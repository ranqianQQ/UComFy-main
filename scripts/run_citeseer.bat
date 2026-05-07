@echo off
cd /d %~dp0\..
python main.py --dataset Citeseer --rewire_method comfy --model UComFyGCN --budget_edges_add 100 --budget_edges_delete 100 --epochs 100 --pretrain_epochs 100 --device cuda
