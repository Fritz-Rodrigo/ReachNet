@echo off
echo Starting ReachNet training...
set PYTHONPATH=%PYTHONPATH%;%CD%\models
set data="C:/Users/rodri/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc/LocalState/rootfs/home/rodrigofritz/f10/500pts"
python training/train_ReachNet.py ^
    --sf_dir "%data%/sf/" ^
    --bup_dir "%data%/bup/" ^
    --sm_stl_dir "%data%/sm/" ^
    --ns_stl_dir "%data%/ns/" ^
    --mesh_type sm ^
    --val_fraction 0.25 ^
    --epochs 10 ^
    --batch_size 2 ^
    --lr 5e-4 ^
    --alpha_geom 1.5
REM "pause" triggers "Press any key to continue . . . " to check outputs & errors
REM pause
