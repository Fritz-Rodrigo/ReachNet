@echo off
echo Starting ReachNet training...
set PYTHONPATH=%PYTHONPATH%;%CD%\models
set data="C:/Users/rodri/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc/LocalState/rootfs/home/rodrigofritz/f10/500pts/v40/"
python training/train_ReachNet.py ^
    --sf_dir "%data%" ^
    --bup_dir "%data%" ^
    --stl_dir "%data%" ^
    --val_fraction 0.25 ^
    --epochs 10 ^
    --batch_size 3 ^
    --lr 5e-4 ^
    --alpha_geom 1.5
REM "pause" triggers "Press any key to continue . . . " to check outputs & errors
REM pause