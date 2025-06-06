from data_prep import data_prep
import os
import glob

# sf_path = "../g7_v100/sf/"
# bup_path = "../g7_v100/bup/"
# stl_path = "../g7_v100/ns/"
# WSL:
sf_path = "C:/Users/rodri/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc/LocalState/rootfs/home/rodrigofritz/f10/"
# Linux:
# sf_path = "../g7/"
bup_path = sf_path
stl_path = sf_path

data = data_prep(sf_path, bup_path, stl_path)

# # Example filename to test parsing
# example_name = "g7_lissajous_3,7,8,0,2pi6,0_16_1.1_±0.2_r0.02_0.029_v90"
# metadata = dummy.parse_metadata(example_name)
# print("Parsed metadata:")
# for key, value in metadata.items():
#     print(f"{key}: {value}")

# Collect correct and wrong parsings
successes = []
failures = []

for fname in os.listdir(sf_path):
    if not fname.endswith("_sf.txt"):
        continue
    base_name = fname.replace("_sf.txt", "")
    pattern = os.path.join(stl_path, base_name + "_a*_ns.stl")  # Match STL with area tag
    matches = glob.glob(pattern)
    if not matches:
        print(f"[Warning] Missing stl for {fname}, skipping.")
        continue
    stl_name = matches[0]  # Use first matching STL
    try:
        metadata = data.parse_metadata_reach_based(stl_name)
        successes.append((stl_name, metadata))
    except Exception as e:
        failures.append((stl_name, str(e)))

print(f"\n✅ Successfully parsed {len(successes)} files.")
print(f"❌ Failed to parse {len(failures)} files.")

# Print and log failures
if failures:
    print("Failures:")
    for f, msg in failures:
        print(f"  {f} → {msg}")

    # Write to log file
    with open("./data/parsing_fail.log", "w") as log_file:
        log_file.write("Failed to parse the following filenames:\n")
        for f, msg in failures:
            log_file.write(f"{f} -> {msg}\n")

    print("\n📝 Failures saved to './data/parsing_fail.log'.")

# Inspect one of the succesfully parsed metadata dictionaries
if successes:
    with open("./data/log_success.txt", "w") as f:
        f.writelines(f"{s}\n" for s in successes)
    
    print("\nExample of parsed metadata:")
    print('File:', successes[0][0])
    for key, val in successes[0][1].items():
        print(f"  {key}: {val}")