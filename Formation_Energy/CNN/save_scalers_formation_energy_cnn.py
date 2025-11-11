import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import joblib
import os
import random

# --- 1. SETUP ---
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# --- 2. DATA LOADING ---
print("Loading data for CNN scalers...")
try:
    df_main = pd.read_csv('6059_data.csv')
    df_magpie = pd.read_csv('magpie_features.csv') 
except FileNotFoundError as e:
    print(f"Error: Could not find data files. {e}")
    exit()

# --- 3. ALIGN, LOAD, AND CLEAN DATA ---
df_main = df_main.dropna(subset=['formation_energy_per_atom']) 
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
        y_list.append(row['formation_energy_per_atom'])
        found_formulas.append(row['formula_sp'])

df_aligned = df_aligned[df_aligned['formula_sp'].isin(found_formulas)].reset_index(drop=True)
X_magpie = df_aligned[magpie_cols].values
X_cnn = np.array(X_cnn_list)
y = np.array(y_list)

# --- 4. OUTLIER REMOVAL (Same as training script) ---
def remove_outliers_advanced(X_cnn_in, X_magpie_in, y_in):
    print(f"Original data shape: {X_cnn_in.shape[0]}")
    Q1, Q3 = np.percentile(y_in, [10, 90]); IQR = Q3 - Q1
    mask1 = (y_in >= (Q1 - 2 * IQR)) & (y_in <= (Q3 + 2 * IQR))
    z_scores = np.abs(stats.zscore(y_in)); mask2 = z_scores < 3.5
    mask = mask1 & mask2
    print(f"Removed {len(y_in) - np.sum(mask)} outliers")
    return X_cnn_in[mask], X_magpie_in[mask], y_in[mask]

X_cnn, X_magpie, y = remove_outliers_advanced(X_cnn, X_magpie, y)
print(f"Final data shape for scaling: {X_cnn.shape}")

# --- 5. FIT AND SAVE SCALERS ---

# 1. CNN Max Value
print("Fitting cnn_max_value...")
cnn_max = np.max(X_cnn)
joblib.dump(cnn_max, 'cnn_max_value_fe.pkl')
print(f"Saved 'cnn_max_value_fe.pkl' (Value: {cnn_max})")

# 2. Magpie Scaler
print("Fitting magpie_scaler...")
scaler_magpie = StandardScaler()
scaler_magpie.fit(X_magpie)
joblib.dump(scaler_magpie, 'magpie_scaler_fe.pkl')
print("Saved 'magpie_scaler_fe.pkl'")

# 3. Target (y) Scaler
print("Fitting y_scaler...")
scaler_y = MinMaxScaler()
scaler_y.fit(y.reshape(-1, 1))
joblib.dump(scaler_y, 'y_scaler_fe.pkl')
print("Saved 'y_scaler_fe.pkl'")

print("\nAll scalers for CNN model saved successfully.")