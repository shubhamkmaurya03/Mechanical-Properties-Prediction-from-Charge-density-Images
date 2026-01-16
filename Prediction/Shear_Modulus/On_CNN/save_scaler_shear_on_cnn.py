import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import joblib
import os
import random
import warnings
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition

# --- 0. SETUP ---
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
warnings.filterwarnings('ignore')
print(" =================== Starting Scaler Generation for 'Best' LGBM Model =================== ")

# --- 1. DATA LOADING ---
print("Loading '6059_rows.csv'...")
try:
    df_main = pd.read_csv('6059_rows.csv')
except FileNotFoundError as e:
    print(f"Error: Could not find data file. {e}")
    exit()

# --- 2. ALIGN, LOAD, AND CLEAN DATA ---
print("Aligning and loading 3D data (this may take a moment)...")
X_3d_list = []
y_list = []
formula_list_pretty = [] # For Magpie generation
formula_list_sp = [] # For Magpie CSV lookup key

input_dir = 'input_cnn'
for index, row in df_main.iterrows():
    file_path = os.path.join(input_dir, f"{row['formula_sp']}_latent.npy")
    if os.path.exists(file_path):
        X_3d_list.append(np.load(file_path))
        y_list.append(row['shear_hill'])
        formula_list_pretty.append(row['formula_pretty'])
        formula_list_sp.append(row['formula_sp'])

X_3d_raw, y = np.array(X_3d_list), np.array(y_list)
print(f"Loaded {X_3d_raw.shape[0]} 3D samples.")

print("Generating Magpie features...")
ep_featurizer = ElementProperty.from_preset("magpie")
try:
    compositions = [Composition(f) for f in formula_list_pretty]
    X_magpie = ep_featurizer.featurize_many(compositions)
    X_magpie = np.array(X_magpie)
    X_magpie = np.nan_to_num(X_magpie, nan=0.0, posinf=0.0, neginf=0.0)
except Exception as e:
    print(f"Error parsing formulas: {e}. Aborting.")
    exit()

# --- Create Magpie Lookup CSV (using formula_sp as key) ---
# This uses the formulas *before* outlier removal for a complete lookup table
magpie_df = pd.DataFrame(X_magpie, columns=ep_featurizer.feature_labels())
magpie_df['formula_sp'] = formula_list_sp # Use the original key
magpie_df.to_csv('magpie_features_shear.csv', index=False)
print("Saved magpie_features_shear.csv (lookup table)")

print("Step 3/5: Applying outlier removal...")
def remove_outliers_advanced(X_3d_in, X_magpie_in, y_in):
    print(f"Original data shape: {X_3d_in.shape[0]}")
    Q1, Q3 = np.percentile(y_in, [10, 90]); IQR = Q3 - Q1
    mask1 = (y_in >= (Q1 - 2 * IQR)) & (y_in <= (Q3 + 2 * IQR))
    z_scores = np.abs(stats.zscore(y_in)); mask2 = z_scores < 3.5
    mask = mask1 & mask2
    print(f"Removed {len(y_in) - np.sum(mask)} outliers")
    return X_3d_in[mask], X_magpie_in[mask], y_in[mask]

X_3d_raw, X_magpie, y = remove_outliers_advanced(X_3d_raw, X_magpie, y)
print(f"Data cleaned. Final sample count: {X_3d_raw.shape[0]}")

# --- 3. FIT AND SAVE ALL SCALERS AND PCA ---
print(" =================== Fitting and saving all preprocessors ===================")

# 1. 3D Data -> Flattened -> Scaled
X_3d_flat = X_3d_raw.reshape(X_3d_raw.shape[0], -1)
scaler_3d = StandardScaler()
X_3d_flat_scaled = scaler_3d.fit_transform(X_3d_flat)
joblib.dump(scaler_3d, 'x_flat_scaler_shear_modulus_best.pkl')
print("Saved x_flat_scaler_shear_modulus_best.pkl")

# 2. PCA on Scaled 3D Data
n_components_pca = 512
pca = PCA(n_components=n_components_pca, random_state=seed_value)
X_pca = pca.fit_transform(X_3d_flat_scaled)
joblib.dump(pca, 'pca_shear_modulus_best.pkl')
print("Saved pca_shear_modulus_best.pkl")

# 3. Magpie Features -> Scaled
scaler_magpie = StandardScaler()
X_magpie_scaled = scaler_magpie.fit_transform(X_magpie)
joblib.dump(scaler_magpie, 'magpie_scaler_shear_modulus_best.pkl')
print("Saved magpie_scaler_shear_modulus_best.pkl")

# 4. Target variable
# NO y_scaler needed for this model, as it trains on raw 'y'
print("Skipping y_scaler (not needed for this model).")

print("\n =================== Done! All 3 scalers for the 'Best' model have been saved. =================== ")