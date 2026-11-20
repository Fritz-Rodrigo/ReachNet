# ReachNet

**A 3D Convolutional Neural Network for Predicting the Reach of Embedded Curves**

## Overview

ReachNet is a deep learning model designed to predict the local reach of an embedded curve from its associated three-dimensional scalar field representation.

The reach of a curve is a geometric quantity related to the largest radius for which its tubular neighborhood remains well-behaved, without self-intersections. Its computation can be challenging, particularly for geometrically complex curves.

ReachNet approaches this problem as a supervised learning task. It takes a 3D scalar field sampled on a regular voxel grid as input and predicts a vector of reach values sampled along the curve.

The architecture combines 3D convolutional layers for volumetric feature extraction with fully connected layers for regression.

## Repository Structure

```text
ReachNet/
│
├── data/
│   ├── data_prep.py
│   ├── data_prep_sm-ns.py
│   ├── parsing_test.py
│   ├── test_data_prep.py
│   ├── test_data_prep_0.py
│   ├── log_success.txt
│   ├── parsing_fail.log
│   └── parsing_failures.log
│
├── models/
│   └── ReachNet.py
│
├── training/
│   ├── train_ReachNet.py
│   ├── train_ReachNet_no_validation.py
│   ├── train_ReachNet_sm-ns.py
│   ├── losses.py
│   └── geometric_losses.py
│
├── utils/
│   ├── bup_to_scalar_field.py
│   ├── marching_cubes.py
│   └── mesh_metrics.py
│
├── tools/
│   └── batch_visualizer.py
│
├── Predictions/
├── Predictions1/
├── Predictions2/
├── Predictions3/
├── ReachPredictions/
├── inference_outputs/
│
├── inference.py
├── eval_pred.py
├── eval_stl.py
├── visualize_predictions.py
├── visualize_predictions_sin.py
│
├── main.py
├── runner.py
│
├── train_config.yaml
├── split_indices.json
├── split_indices1.json
├── split_indices2.json
├── split_indices3.json
│
├── run_train_ReachNet.bat
├── run_train_ReachNet_sm-ns.bat
├── run_train_sinusoidal_ReachNet.bat
│
└── ReachNet_*.pth
```

## Architecture

ReachNet uses a 3D CNN to process volumetric scalar-field representations.

The network consists of:

1. **3D convolutional feature extractor:** successive convolutional layers learn spatial and geometric features from the input volume.
2. **Pooling:** an adaptive pooling operation produces a fixed-size feature representation.
3. **Fully connected regression head:** an MLP maps the extracted features to the predicted reach profile.

The overall architecture can be summarized as:

```text
3D Scalar Field
      |
      v
  3D Convolutional Layers
      |
      v
  Feature Extraction
      |
      v
 Adaptive Average Pooling
      |
      v
   Flattening
      |
      v
 Fully Connected Layers
      |
      v
 Predicted Reach Profile
```

The convolutional layers exploit local spatial structure and weight sharing, while the regression head learns the mapping between the volumetric representation and the reach values along the curve.

## Data Pipeline

The `data/` directory contains the scripts responsible for preparing the training data.

- `data_prep.py`: primary data preparation and parsing pipeline.
- `data_prep_sm-ns.py`: alternative data preparation pipeline for the smoothed/non-smoothed dataset configuration.
- `parsing_test.py`: parsing tests.
- `test_data_prep.py`, `test_data_prep_0.py`: data preparation tests.

The `utils/` directory contains auxiliary geometric transformations:

- `bup_to_scalar_field.py`: conversion between BUP radius profiles and scalar-field representations.
- `marching_cubes.py`: Marching Cubes utility.
- `mesh_metrics.py`: mesh-related metrics.

## Training

The `training/` directory contains the model optimization routines and loss functions.

- `train_ReachNet.py`: primary training implementation.
- `train_ReachNet_no_validation.py`: training implementation without a validation stage.
- `train_ReachNet_sm-ns.py`: alternative training implementation for the smoothed/non-smoothed data configuration.
- `losses.py`: loss functions.
- `geometric_losses.py`: geometry-oriented loss functions.

Training configuration is maintained in `train_config.yaml`.

The repository also contains Windows batch scripts for launching different training configurations:

```powershell
.\run_train_ReachNet.bat
.\run_train_ReachNet_sm-ns.bat
.\run_train_sinusoidal_ReachNet.bat
```

Dataset split indices are stored in the corresponding `split_indices*.json` files.

## Model Checkpoints

Trained model weights are stored as PyTorch `.pth` files in the repository root.

These include:

- `ReachNet_15epochs.pth`
- `ReachNet_20epochs.pth`
- `ReachNet_25epochs.pth`
- `ReachNet_30epochs.pth`
- `ReachNet_50epochs.pth`
- `trained_ReachNet_sin1.pth`
- `trained_ReachNet_sin2.pth`
- `trained_ReachNet_sin3.pth`
- `trained_ReachNet_sin4.pth`

These checkpoints correspond to different training runs and experimental configurations.

## Inference

Inference is implemented in `inference.py`.

The model receives a prepared scalar-field volume and produces a predicted reach profile.

```powershell
python inference.py
```

The repository also includes `main.py` and `runner.py` as execution entry points.

## Evaluation

The repository provides scripts for evaluating predictions and their geometric consequences.

- `eval_pred.py`: evaluation of predicted reach profiles.
- `eval_stl.py`: evaluation involving STL meshes.
- `visualize_predictions.py`: visualization of predicted results.
- `visualize_predictions_sin.py`: visualization for sinusoidal experiments.

The `tools/` directory includes `batch_visualizer.py` for batch visualization.

## Prediction Outputs

Several directories contain results from different experimental runs:

| Directory | Contents |
|---|---|
| `Predictions/` | Predicted and reference reach profiles with plots. |
| `Predictions1/` | Earlier prediction experiments, including generated animations. |
| `Predictions2/` | Predictions for selected geometric examples, including STL-related outputs. |
| `Predictions3/` | Additional prediction experiments. |
| `ReachPredictions/` | Predictions for selected higher-genus examples. |
| `inference_outputs/` | Inference results. |

These directories preserve experimental outputs and visual comparisons between predicted and reference reach profiles.

## Requirements

ReachNet is implemented in Python and uses PyTorch for deep learning.

The computational environment requires:

- Python
- PyTorch
- NumPy
- SciPy
- Matplotlib
- scikit-image

Additional dependencies may be required by the data preparation, visualization, and mesh-processing utilities.

A CUDA-capable NVIDIA GPU is recommended for accelerating 3D convolutional training and inference.

## Acknowledgments

The training samples used in ReachNet were generated using [EuLearn](https://github.com/appliedgeometry/EuLearn_db), an open-source software project distributed under the GNU General Public License v3.0.

## License

ReachNet is distributed under the [MIT License](https://opensource.org/license/mit).
