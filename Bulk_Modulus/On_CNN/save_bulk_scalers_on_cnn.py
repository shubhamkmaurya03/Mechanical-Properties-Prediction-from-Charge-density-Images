import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import joblib
import os
import random
import warnings

warnings.filterwarnings('ignore')

# --- 0. SETUP ---
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# --- 1. DATA LOADING ---
print(" ==================== Loading data to generate LGBM (on_cnn) scalers ==================== ")
try:
    df_main = pd.read_csv('6059_rows.csv')
    df_magpie = pd.read_csv('magpie_features.csv') 
    print("Successfully loaded '6059_rows.csv' and 'magpie_features.csv'")
except FileNotFoundError as e:
    print(f"Error: Could not find data files. {e}")
    exit()

# --- 2. ALIGN, LOAD, AND CLEAN DATA ---
print(" =================== Aligning and cleaning data =================== ")
df_main = df_main.dropna(subset=['bulk_hill'])
df_aligned = pd.merge(df_main, df_magpie, on='formula_sp', how='inner')
magpie_cols = [col for col in df_magpie.columns if col != 'formula_sp']

X_cnn_list = []
y_list = []
input_dir = 'input_cnn'
found_formulas = []
for index, row in df_aligned.iterrows():
    file_path = os.path.join(input_dir, f"{row['formula_sp']}_latent.npy")
    if os.path.exists(file_path):
        X_cnn_list.append(np.load(file_path))
        y_list.append(row['bulk_hill'])
        found_formulas.append(row['formula_sp'])

df_aligned = df_aligned[df_aligned['formula_sp'].isin(found_formulas)].reset_index(drop=True)
X_magpie_unscaled = df_aligned[magpie_cols].values
X_3d_raw = np.array(X_cnn_list)
y = np.array(y_list)

# --- 3. APPLY *EXACT* SAME OUTLIER REMOVAL (from V6 script) ---
def remove_outliers_iqr(X_3d_in, X_magpie_in, y_in):
    print(f"Original data shape: {X_3d_in.shape[0]}")
    finite_mask = np.isfinite(y_in)
    X_3d_in, X_magpie_in, y_in = X_3d_in[finite_mask], X_magpie_in[finite_mask], y_in[finite_mask]
    
    Q1 = np.percentile(y_in, 25); Q3 = np.percentile(y_in, 75); IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
    
    mask = (y_in >= lower_bound) & (y_in <= upper_bound)
    print(f"Removed {len(y_in) - np.sum(mask)} numerical outliers (1.5*IQR)")
    return X_3d_in[mask], X_magpie_in[mask], y_in[mask]

X_3d_final, X_magpie_final, y_final = remove_outliers_iqr(X_3d_raw, X_magpie_unscaled, y)
print(f"Data cleaned. Final shape: {X_3d_final.shape}")

# --- 4. FIT AND SAVE THE SCALERS/PIPELINE ---
print(" ==================== Fitting and saving scalers ==================== ")

# 1. Fit and Save the 3D Scaler
X_3d_flat = X_3d_final.reshape(X_3d_final.shape[0], -1)
scaler_3d = StandardScaler()
X_3d_flat_scaled = scaler_3d.fit_transform(X_3d_flat)
joblib.dump(scaler_3d, 'bulk_pca_scaler_on_cnn.pkl')
print("Saved bulk_pca_scaler_on_cnn.pkl")

# 2. Fit and Save the PCA object
n_components_pca = 512 # This matches the V6 script
pca = PCA(n_components=n_components_pca, random_state=seed_value)
pca.fit(X_3d_flat_scaled)
joblib.dump(pca, 'bulk_pca_model_on_cnn.pkl')
print(f"Saved bulk_pca_model_on_cnn.pkl (n_components={n_components_pca})")

# 3. Fit and Save the Magpie Scaler
scaler_magpie = StandardScaler()
scaler_magpie.fit(X_magpie_final)
joblib.dump(scaler_magpie, 'bulk_magpie_scaler_on_cnn.pkl')
print("Saved bulk_magpie_scaler_on_cnn.pkl")

print("\nDone! All scalers ('keys') for the LGBM (on_cnn) model have been saved.")