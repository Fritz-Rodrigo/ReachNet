# runner.py
from inference import run_inference
import os

if __name__ == "__main__":
    sf_path = "../g7_v100/sf/g7_lissajous_4,5,7,0,0,pi3_03_1.1_±0.2_r0.02_0.029_v90_sf.txt"
    model_path = "trained_model.pth"
    output_dir = "inference_outputs"

    os.makedirs(output_dir, exist_ok=True)
    run_inference(sf_path, model_path, output_dir)