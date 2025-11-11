import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
import joblib
import os
import random
import warnings

# --- Import Matminer ---
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition

# --- 0. SETUP ---
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
warnings.filterwarnings('ignore')
print(" =================== Starting Scaler Generation for 3-Branch CNN Model =================== ")

# --- 1. DATA LOADING ---
print("Loading '6059_rows.csv'...")
try:
    df_main = pd.read_csv('6059_rows.csv')
except FileNotFoundError as e:
    print(f"Error: Could not find data file. {e}")
    exit()

# --- 2. ALIGN, LOAD, AND CLEAN DATA ---
print("Aligning and loading 3D data (this may take a moment)...")
X_3d_data, y_data, formula_list_pretty, formula_list_sp = [], [], [], []
input_dir = 'input_cnn' # ASSUMES 16x16x16 data
for _, row in df_main.iterrows():
    file_path = os.path.join(input_dir, f"{row['formula_sp']}_latent.npy")
    if os.path.exists(file_path):
        X_3d_data.append(np.load(file_path))
        y_data.append(row['shear_hill']) # <-- Target is shear_hill
        formula_list_pretty.append(row['formula_pretty'])
        formula_list_sp.append(row['formula_sp']) # Keep original key for lookup
        
X, y = np.array(X_3d_data), np.array(y_data)
print(f"Loaded {X.shape[0]} 3D samples.")

# --- 3. APPLY *EXACT* SAME OUTLIER REMOVAL ---
def remove_outliers_advanced(X, y, formulas, formula_sps):
    Q1, Q3 = np.percentile(y, [10, 90]); IQR = Q3 - Q1
    mask1 = (y >= (Q1 - 2 * IQR)) & (y <= (Q3 + 2 * IQR))
    z_scores = np.abs(stats.zscore(y)); mask2 = z_scores < 3.5
    mask = mask1 & mask2
    print(f"Removed {len(y) - np.sum(mask)} outliers")
    formulas_out = [formulas[i] for i in range(len(formulas)) if mask[i]]
    formula_sps_out = [formula_sps[i] for i in range(len(formula_sps)) if mask[i]]
    return X[mask], y[mask], formulas_out, formula_sps_out

X_final, y_final, formula_list_final, formula_sp_list_final = remove_outliers_advanced(X, y, formula_list_pretty, formula_list_sp)
print(f"Data cleaned. Final sample count: {X_final.shape[0]}")

# --- 4. FIT AND SAVE ALL SCALERS AND PCA ---
print(" =================== Fitting and saving all preprocessors ===================")

# 1. Save the 3D CNN Scaler (the global max value)
cnn_max_val = np.max(X_final)
joblib.dump(cnn_max_val, 'cnn_max_value_shear_modulus_cnn.pkl')
print(f"Saved cnn_max_value_shear_modulus_cnn.pkl (Value: {cnn_max_val})")

# 2. Save the Target (y) Scaler (RobustScaler as used in training)
scaler_y = RobustScaler()
scaler_y.fit(y_final.reshape(-1, 1))
joblib.dump(scaler_y, 'y_scaler_shear_modulus_cnn.pkl')
print("Saved y_scaler_shear_modulus_cnn.pkl")

# 3. Save the Magpie Scaler
print("Generating Magpie features...")
ep_featurizer = ElementProperty.from_preset("magpie")
compositions = [Composition(f) for f in formula_list_final]
X_magpie = ep_featurizer.featurize_many(compositions)
X_magpie = np.array(X_magpie)
X_magpie = np.nan_to_num(X_magpie, nan=0.0, posinf=0.0, neginf=0.0)

scaler_magpie = StandardScaler()
X_magpie_scaled = scaler_magpie.fit_transform(X_magpie)
joblib.dump(scaler_magpie, 'magpie_scaler_shear_modulus_cnn.pkl')
print("Saved magpie_scaler_shear_modulus_cnn.pkl")

# 4. Save the Magpie Feature Lookup Table (using formula_sp as the key)
magpie_df = pd.DataFrame(X_magpie, columns=ep_featurizer.feature_labels())
magpie_df['formula_sp'] = formula_sp_list_final # Use the original key
magpie_df.to_csv('magpie_features_shear.csv', index=False)
print("Saved magpie_features_shear.csv (lookup table)")

# 5. Save the 3D-Flattened Scaler (for PCA)
X_flat = X_final.reshape(X_final.shape[0], -1)
scaler_3d_flat = StandardScaler()
X_flat_scaled = scaler_3d_flat.fit_transform(X_flat)
joblib.dump(scaler_3d_flat, 'x_flat_scaler_shear_modulus_cnn.pkl')
print("Saved x_flat_scaler_shear_modulus_cnn.pkl")

# 6. Save the PCA Model
n_components_pca = 512 # Must match the n_components used in training
pca = PCA(n_components=n_components_pca, random_state=seed_value)
pca.fit(X_flat_scaled)
joblib.dump(pca, 'pca_shear_modulus_cnn.pkl')
print("Saved pca_shear_modulus_cnn.pkl")

print("\n =================== Done! All 6 assets for the CNN model have been saved. =================== ")