@echo off
cd /d %~dp0\..
python main.py --dataset Pubmed --rewire_method comfy --model UComFyGCN --budget_edges_add 300 --budget_edges_delete 300 --epochs 100 --pretrain_epochs 100 --device cuda
