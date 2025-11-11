import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
print(" =================== Loading data to generate Keras scalers =================== ")
try:
    df_main = pd.read_csv('6059_rows.csv')
    df_magpie = pd.read_csv('magpie_features.csv') 
    print("Successfully loaded '6059_rows.csv' and 'magpie_features.csv'")
except FileNotFoundError as e:
    print(f"Error: Could not find data files. {e}")
    exit()

# --- 2. ALIGN, LOAD, AND CLEAN DATA ---
print(" =================== Aligning and cleaning data (this may take a moment) ===================")
df_main = df_main.dropna(subset=['bulk_hill'])
df_aligned = pd.merge(df_main, df_magpie, on='formula_sp', how='inner')
magpie_cols = [col for col in df_magpie.columns if col != 'formula_sp']

X_cnn_list = []
y_list = []
X_magpie_list = []
input_dir = 'input_cnn'
found_formulas = []
for index, row in df_aligned.iterrows():
    file_path = os.path.join(input_dir, f"{row['formula_sp']}_latent.npy")
    if os.path.exists(file_path):
        X_cnn_list.append(np.load(file_path))
        # --- MODIFIED FOR BULK ---
        y_list.append(row['bulk_hill'])
        X_magpie_list.append(row[magpie_cols].values)

X_cnn = np.array(X_cnn_list)
X_magpie = np.array(X_magpie_list)
y = np.array(y_list)

# --- 3. APPLY *EXACT* SAME OUTLIER REMOVAL (from V7 script) ---
def remove_outliers_advanced(X_cnn_in, X_magpie_in, y_in):
    y_finite_mask = np.isfinite(y_in)
    X_cnn_in = X_cnn_in[y_finite_mask]; X_magpie_in = X_magpie_in[y_finite_mask]; y_in = y_in[y_finite_mask]
    Q1 = np.percentile(y_in, 25); Q3 = np.percentile(y_in, 75); IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (y_in >= lower_bound) & (y_in <= upper_bound)
    print(f"Original data shape: {len(y_in)}")
    print(f"Removed {len(y_in) - np.sum(outlier_mask)} outliers using 1.5*IQR")
    y_filtered = y_in[outlier_mask]
    if len(y_filtered) == 0 or np.std(y_filtered) == 0:
        return X_cnn_in, X_magpie_in, y_in
    else:
        return X_cnn_in[outlier_mask], X_magpie_in[outlier_mask], y_in[outlier_mask]

X_cnn_final, X_magpie_final, y_final = remove_outliers_advanced(X_cnn, X_magpie, y)
print(f"Data cleaned. Final shape: {X_cnn_final.shape}")

# --- 4. FIT AND SAVE THE SCALERS ---
print(" =================== Fitting and saving scalers ===================")

# 1. Save the 3D CNN Scaler (the global max value)
cnn_max_val = np.max(X_cnn_final)
joblib.dump(cnn_max_val, 'bulk_cnn_max_value.pkl')
print(f"Saved bulk_cnn_max_value.pkl (Value: {cnn_max_val})")

# 2. Save the Magpie Scaler
scaler_magpie = StandardScaler()
scaler_magpie.fit(X_magpie_final)
joblib.dump(scaler_magpie, 'bulk_magpie_scaler.pkl')
print("Saved bulk_magpie_scaler.pkl")

# 3. Save the Target (y) Scaler
scaler_y = MinMaxScaler() 
scaler_y.fit(y_final.reshape(-1, 1))
joblib.dump(scaler_y, 'bulk_y_scaler.pkl')
print("Saved bulk_y_scaler.pkl")

print("\n =================== Done! All Keras scalers ('keys') have been saved =================== ")