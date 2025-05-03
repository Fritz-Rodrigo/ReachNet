@echo off
echo Starting ReachNet training...
set PYTHONPATH=%PYTHONPATH%;%CD%\models  # Add models folder to PYTHONPATH
python training/train_ReachNet.py ^
    --sf_dir ../g7_v100/sf ^
    --bup_dir ../g7_v100/bup ^
    --stl_dir ../g7_v100/ns ^
    --epochs 10 ^
    --batch_size 2 ^
    --lr 5e-4 ^
    --alpha_geom 1.5
pause
