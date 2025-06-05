import os
import glob
import re
from data_prep import data_prep

sf_dir = r"C:/Users/rodri/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc/LocalState/rootfs/home/rodrigofritz/f10/500pts/sf"
bup_dir = r"C:/Users/rodri/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc/LocalState/rootfs/home/rodrigofritz/f10/500pts/bup"
sm_stl_dir = r"C:/Users/rodri/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc/LocalState/rootfs/home/rodrigofritz/f10/500pts/sm"
ns_stl_dir = r"C:/Users/rodri/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc/LocalState/rootfs/home/rodrigofritz/f10/500pts/ns"
mesh_type = "sm"

dataset = data_prep(sf_dir, bup_dir, sm_stl_dir, ns_stl_dir, mesh_type=mesh_type)

print("\n============================")
print(f"✅ Total examples loaded: {len(dataset)}")
print("============================\n")

for i, entry in enumerate(dataset.files[:10]):
    sf_path, bup_path, sm_stl_path, ns_stl_path, base_name = entry
    print(f"[{i}] ✅ Loaded: {base_name}")
    print(f"     - Scalar Field: {os.path.basename(sf_path)}")
    print(f"     - Blow-up:      {os.path.basename(bup_path)}")
    print(f"     - STL (sm):     {os.path.basename(sm_stl_path) if sm_stl_path else 'None'}")
    print(f"     - STL (ns):     {os.path.basename(ns_stl_path) if ns_stl_path else 'None'}")
    print()

if len(dataset) == 0:
    print("🚨 No examples were loaded.")
    print("🔍 Check filenames, area tag in STL, directory paths, or mesh_type filter.")
