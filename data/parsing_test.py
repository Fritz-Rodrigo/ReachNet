from data_prep import data_prep

sf_path = "../g7_v100/sf/"
bup_path = "../g7_v100/bup/"
stl_path = "../g7_v100/ns/"

# Example filename to test
example_name = "g7_lissajous_3,7,8,0,pi5,0_16_1.1_±0.2_r0.02_0.029_v90"

# Create dummy dataset just to access the parsing method
dummy = data_prep(sf_path, bup_path, stl_path)
metadata = dummy.parse_metadata(example_name)

print("✅ Parsed metadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")