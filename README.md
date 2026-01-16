# Mechanical Properties Prediction from Charge Density Images using CNN and MAGPIE Features

A machine learning project that predicts five mechanical properties of materials from charge density images using Convolutional Neural Networks (CNN) combined with MAGPIE (Materials Agnostic General-Purpose Interface Energy) features.

## Project Overview

This project develops deep learning models to predict critical mechanical properties of crystalline materials:
- **Bulk Modulus**: Resistance to uniform compression
- **Debye Temperature**: Characteristic temperature related to material stiffness
- **Formation Energy Per Atom**: Energy required to form the material
- **Shear Modulus**: Resistance to shear deformation
- **Young's Modulus**: Tensile stiffness of the material

The prediction combines:
1. **Charge density images** processed through a CNN architecture with attention mechanisms
2. **Autoencoder compression** (8, 16, and 32-dimensional representations)
3. **MAGPIE elemental features** for chemical property encoding

## Project Structure

```
├── Charge_density_compression_AE/          # Autoencoder models for charge density compression
│   ├── AE_8/                               # 8-dimensional latent space
│   ├── AE_16/                              # 16-dimensional latent space
│   └── AE_32/                              # 32-dimensional latent space
│
├── Prediction/                              # Property prediction models
│   ├── Bulk_Modulus/
│   ├── Debye_Temperature/
│   ├── Formation_Energy/
│   ├── Shear_Modulus/
│   └── Youngs_Modulus/
│       ├── CNN/                            # Convolutional Neural Network models
│       └── On_CNN/                         # Fusion models (CNN + MAGPIE features)
```

## Installation

### Requirements
- Python 3.10+
- TensorFlow/Keras
- NumPy, Pandas, Scikit-learn
- Joblib, Matplotlib

### Setup Environment

Each property prediction folder contains an `environment.yml` file for creating a conda environment:

```bash
cd Prediction/[Property]/CNN
conda env create -f environment.yml
conda activate new-gpu-env
```

## Usage

### 1. Autoencoder Compression

First, compress charge density images using autoencoders:

```bash
cd Charge_density_compression_AE/AE_8  # or AE_16, AE_32
jupyter notebook 128_ae_8.ipynb
```

This produces latent representations (`.npy` files) of the charge density images.

### 2. Predict Mechanical Properties

For any mechanical property (e.g., Bulk Modulus):

```bash
cd Prediction/Bulk_Modulus/CNN
python predict_bulk.py
```

**Required files in the directory:**
- Trained model: `To_Publish_Bulk_Modulus_Best.h5`
- Custom layers: `custom_layers_bulk.py`
- Scalers: `bulk_cnn_max_value.pkl`, `bulk_magpie_scaler.pkl`, `bulk_y_scaler.pkl`
- MAGPIE features: `magpie_features.csv`
- Input: Compressed charge density `.npy` files

**Input format:**
```python
formula = "Ac2CuGe_225"
npy_file_path = "path/to/compressed_density_latent.npy"
```

### 3. Using Fusion Models (CNN + MAGPIE)

For improved predictions using combined CNN and elemental features:

```bash
cd Prediction/[Property]/On_CNN
python predict_[property]_on_cnn.py
```

## Model Architecture

### Autoencoder for Charge Density
- Input: 3D charge density matrices
- Encoder: Convolutional layers reducing spatial dimensions
- Latent space: 8, 16, or 32 dimensions
- Decoder: Reconstructs charge density from latent representation

### CNN with Attention Mechanism
- Custom `DualAttention3D` layer for spatial and channel attention
- Processes compressed charge density representations
- Outputs normalized predictions for mechanical properties

### Fusion Model (On_CNN)
- Concatenates CNN features with MAGPIE elemental features
- Fully connected layers for final prediction
- Enhanced prediction accuracy by combining image and chemical information

## Data

Each property folder contains:
- `6059_rows.csv` or `6059_data.csv`: Full dataset
- `train_split.csv`, `val_split.csv`, `test_split.csv`: Data splits
- `magpie_features.csv`: Pre-computed MAGPIE features for elements

## Key Files

| File | Purpose |
|------|---------|
| `custom_layers_[property].py` | Custom Keras/TensorFlow layers (DualAttention3D) |
| `predict_[property].py` | Prediction script for CNN model |
| `predict_[property]_on_cnn.py` | Prediction script for fusion model |
| `save_scalers_[property].py` | Script to save/load data scalers |
| `To_Publish_[Property]_Best.h5` | Trained model weights |

## Scaling and Normalization

The models use three scalers:
1. **CNN Max Value Scaler**: Normalizes charge density image values
2. **MAGPIE Scaler**: Normalizes elemental features
3. **Y Scaler**: Inverse-transforms predictions to actual property values

Load scalers with joblib:
```python
import joblib
scaler = joblib.load('bulk_cnn_max_value.pkl')
```

## Results

Models are trained on 6,059 materials with validated accuracy on:
- Training set (80%)
- Validation set (10%)
- Test set (10%)

Each property has:
- CNN-only model
- Fusion model (CNN + MAGPIE features)

The fusion approach typically provides 5-15% improvement over CNN-only predictions.

## Environment Variables

The prediction scripts set TensorFlow logging to suppress verbose output:
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```
