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
print(" =================== Loading data to generate DEBYE scalers =================== ")
try:
    df_main = pd.read_csv('6059_rows.csv')
    df_magpie = pd.read_csv('magpie_features.csv') 
    print("Successfully loaded '6059_rows.csv' and 'magpie_features.csv'")
except FileNotFoundError as e:
    print(f"Error: Could not find data files. {e}")
    exit()

# --- 2. ALIGN, LOAD, AND CLEAN DATA ---
print(" =================== Aligning and cleaning data (this may take a moment) ===================")
# CHANGED: Target variable to debye_temperature
df_main = df_main.dropna(subset=['debye_temperature'])
df_aligned = pd.merge(df_main, df_magpie, on='formula_sp', how='inner')
magpie_cols = [col for col in df_magpie.columns if col != 'formula_sp']
X_magpie_unscaled = df_aligned[magpie_cols].values

X_cnn_list = []
y_list = []
input_dir = 'input_cnn'
found_formulas = []
for index, row in df_aligned.iterrows():
    file_path = os.path.join(input_dir, f"{row['formula_sp']}_latent.npy")
    if os.path.exists(file_path):
        X_cnn_list.append(np.load(file_path))
        # CHANGED: Target variable to debye_temperature
        y_list.append(row['debye_temperature'])
        found_formulas.append(row['formula_sp'])

df_aligned = df_aligned[df_aligned['formula_sp'].isin(found_formulas)].reset_index(drop=True)
X_magpie = df_aligned[magpie_cols].values
X_cnn = np.array(X_cnn_list)
y = np.array(y_list)

# --- 3. APPLY *EXACT* SAME OUTLIER REMOVAL (from v18/fine-tune script) ---
def remove_outliers_advanced(X_cnn_in, X_magpie_in, y_in):
    y_finite_mask = np.isfinite(y_in)
    X_cnn_in = X_cnn_in[y_finite_mask]; X_magpie_in = X_magpie_in[y_finite_mask]; y_in = y_in[y_finite_mask]
    Q1 = np.percentile(y_in, 25); Q3 = np.percentile(y_in, 75); IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (y_in >= lower_bound) & (y_in <= upper_bound)
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
# CHANGED: Filename
joblib.dump(cnn_max_val, 'cnn_max_value_debye.pkl')
print(f"Saved cnn_max_value_debye.pkl (Value: {cnn_max_val})")

# 2. Save the Magpie Scaler
scaler_magpie = StandardScaler()
scaler_magpie.fit(X_magpie_final)
# CHANGED: Filename
joblib.dump(scaler_magpie, 'magpie_scaler_debye.pkl')
print("Saved magpie_scaler_debye.pkl")

# 3. Save the Target (y) Scaler
scaler_y = MinMaxScaler() 
scaler_y.fit(y_final.reshape(-1, 1))
# CHANGED: Filename
joblib.dump(scaler_y, 'y_scaler_debye.pkl')
print("Saved y_scaler_debye.pkl")

print("\n =================== Done! All DEBYE scalers ('keys') have been saved =================== ")