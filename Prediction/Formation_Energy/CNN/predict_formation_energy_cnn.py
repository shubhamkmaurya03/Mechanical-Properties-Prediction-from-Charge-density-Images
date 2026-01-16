import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import pandas as pd
import joblib
import warnings
import os
import sys
import re

# --- NEW IMPORTS FOR FEATURE GENERATION ---
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

# --- 0. SETUP ---
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow warnings

# CRITICAL: Import the custom layer definition
try:
    from custom_layers_fe import DualAttention3D
except ImportError:
    print("--- FATAL ERROR: 'custom_layers_fe.py' not found. ---")
    print("Please make sure 'custom_layers_fe.py' is in the same directory.")
    sys.exit(1)

# --- 1. HELPER: DYNAMIC FEATURE GENERATION ---

def generate_missing_magpie_features(formula_input, magpie_csv_path, assets_df):
    """
    Generates Magpie features for a missing formula, appends to CSV, 
    and returns the single row DataFrame.
    """
    print(f"   [Info] Magpie features missing for '{formula_input}'. Generating now...")
    
    # 1. Prepare Data
    # Clean the formula (remove _xxx suffix if present) just like in magpieGenerate.py
    formula_clean = re.sub(r'_\d+$', '', formula_input)
    
    # Create a temporary DataFrame
    temp_df = pd.DataFrame({
        'formula_sp': [formula_input], 
        'formula_clean': [formula_clean]
    })

    try:
        # 2. Convert to Composition
        str_to_comp = StrToComposition(target_col_id='composition')
        df_comp = str_to_comp.featurize_dataframe(temp_df, col_id='formula_clean', ignore_errors=False, verbose=False)
        
        # 3. Generate Features
        ep_feat = ElementProperty.from_preset("magpie")
        # Featurize
        df_features = ep_feat.featurize_dataframe(df_comp, col_id='composition', ignore_errors=False, verbose=False)
        
        # 4. Filter Columns
        # We need to ensure we only keep columns that match the loaded assets
        # (excluding formula_sp which is the key)
        required_cols = [col for col in assets_df.columns if col != 'formula_sp']
        
        # Check if generation failed (sometimes matminer returns NaNs for invalid elements)
        if df_features[required_cols].isnull().values.any():
            print(f"   [Error] Generated features contain NaNs for {formula_clean}.")
            return None

        # Prepare final row (formula_sp + features)
        new_row = df_features[['formula_sp'] + required_cols]

        # 5. Save to CSV (Append mode)
        # Check if file exists to determine if we need header
        header = not os.path.exists(magpie_csv_path)
        new_row.to_csv(magpie_csv_path, mode='a', header=header, index=False)
        print(f"   [Success] Features generated and appended to {magpie_csv_path}")

        return new_row

    except Exception as e:
        print(f"   [Error] Failed to generate features: {e}")
        return None

# --- 2. DEFINE PREDICTION FUNCTIONS ---

def load_prediction_assets_cnn():
    """Loads all necessary files (model, scalers, lookup table)."""
    assets = {}
    try:
        # Register the custom layer
        tf.keras.utils.get_custom_objects()['DualAttention3D'] = DualAttention3D
        
        # Load the trained model
        assets['model'] = models.load_model('To_Publish_Formation_Energy_Best_CNN.h5')
        
        # Load the scalers (the "keys")
        assets['cnn_max_val'] = joblib.load('cnn_max_value_fe.pkl')
        assets['magpie_scaler'] = joblib.load('magpie_scaler_fe.pkl')
        assets['y_scaler'] = joblib.load('y_scaler_fe.pkl')
        
        # Load the Magpie feature lookup table
        assets['magpie_csv_path'] = 'magpie_features.csv' # Store path for appending later
        assets['magpie_lookup_df'] = pd.read_csv(assets['magpie_csv_path'])
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        print("All 3D-CNN model assets loaded successfully.")
        
    except FileNotFoundError as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Missing file: {e.filename}")
        print("Please ensure all files are in the same directory:")
        print("[To_Publish_Formation_Energy_Best_CNN.h5, custom_layers_fe.py, cnn_max_value_fe.pkl, magpie_scaler_fe.pkl, y_scaler_fe.pkl, magpie_features.csv]")
        return None
    except Exception as e:
        print(f"An unknown error occurred loading assets: {e}")
        return None
    return assets

def predict_formation_energy_cnn(formula: str, npy_file_path: str, assets: dict) -> float:
    """
    Predicts the Formation Energy (eV/atom) from a formula and a .npy file.
    """
    if not assets:
        print("Error: Model assets are not loaded.")
        return None
        
    print(f"\n--- Predicting for {formula} ---")
    
    # === A: PROCESS 3D CNN INPUT ===
    try:
        x_3d = np.load(npy_file_path)
    except FileNotFoundError:
        print(f"Error: Cannot find file {npy_file_path}")
        return None
        
    x_3d_scaled = x_3d / assets['cnn_max_val']
    x_3d_final = np.expand_dims(x_3d_scaled, axis=0) # Add batch dimension

    # === B: PROCESS MAGPIE INPUT ===
    try:
        # 1. Try to find features in loaded DataFrame
        x_1d_features = assets['magpie_lookup_df'][assets['magpie_lookup_df']['formula_sp'] == formula][assets['magpie_cols']]
        
        # 2. If not found, GENERATE THEM
        if x_1d_features.empty:
            new_row = generate_missing_magpie_features(formula, assets['magpie_csv_path'], assets['magpie_lookup_df'])
            
            if new_row is not None:
                # Use the newly generated row
                x_1d_features = new_row[assets['magpie_cols']]
                
                # OPTIONAL: Update the in-memory dataframe so subsequent calls don't re-generate
                assets['magpie_lookup_df'] = pd.concat([assets['magpie_lookup_df'], new_row], ignore_index=True)
            else:
                print(f"Error: Could not generate Magpie features for '{formula}'")
                return None

    except Exception as e:
        print(f"Error looking up/generating Magpie features: {e}")
        return None
    
    # Scale features
    x_1d_scaled = assets['magpie_scaler'].transform(x_1d_features)

    # === C: MAKE PREDICTION ===
    model_inputs = {
        "cnn_input": x_3d_final, 
        "magpie_input": x_1d_scaled
    }
    
    try:
        y_pred_scaled = assets['model'].predict(model_inputs, verbose=0)
        y_pred_real = assets['y_scaler'].inverse_transform(y_pred_scaled)
        final_prediction = y_pred_real[0][0]
        return final_prediction
        
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

# --- 3. EXAMPLE USAGE ---
if __name__ == "__main__":
    
    # Load all assets once at the start
    model_assets = load_prediction_assets_cnn()
    
    if model_assets:
        # --- Example: Handle user input (CLI) ---
        print("\n" + "="*50)
        print("Usage: python predict.py <formula_string> <path_to_npy_file>")
        print("If formula features are missing, they will be auto-generated.")
        
        if len(sys.argv) == 3:
            formula_from_gui = sys.argv[1]
            npy_path_from_gui = sys.argv[2]
            
            prediction_gui = predict_formation_energy_cnn(
                formula=formula_from_gui,
                npy_file_path=npy_path_from_gui,
                assets=model_assets
            )
            if prediction_gui is not None:
                print(f"-> Final Prediction for {formula_from_gui} is {prediction_gui:.4f} eV/atom")
        else:
            # Fallback test if no args provided
            print("No arguments provided. Running test on loaded data...")
            if not model_assets['magpie_lookup_df'].empty:
                test_formula = model_assets['magpie_lookup_df'].iloc[0]['formula_sp']
                test_npy_file = f"input_cnn/{test_formula}_latent.npy" # Assuming structure
                
                # Fake a "missing" file test by using a made-up formula
                # fake_formula = "NaCl_999" 
                # prediction = predict_formation_energy_cnn(fake_formula, test_npy_file, model_assets)
            
        print("="*50)