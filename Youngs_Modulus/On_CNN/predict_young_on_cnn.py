import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
import sys
import warnings
import re

# --- NEW IMPORTS FOR FEATURE GENERATION ---
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

# --- 0. SETUP ---
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# --- 1. HELPER: DYNAMIC FEATURE GENERATION ---

def generate_missing_magpie_features(formula_input, magpie_csv_path, assets_df):
    """
    Generates Magpie features for a missing formula, appends to CSV, 
    and returns the single row DataFrame.
    """
    print(f"   [Info] Magpie features missing for '{formula_input}'. Generating now...")
    
    # 1. Prepare Data
    # Clean the formula (remove _xxx suffix if present)
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
        # Ensure we only keep columns that match the loaded assets (excluding formula_sp)
        required_cols = [col for col in assets_df.columns if col != 'formula_sp']
        
        # Check if generation failed (sometimes matminer returns NaNs for invalid elements)
        if df_features[required_cols].isnull().values.any():
            print(f"   [Error] Generated features contain NaNs for {formula_clean}.")
            return None

        # Prepare final row
        new_row = df_features[['formula_sp'] + required_cols]

        # 5. Save to CSV (Append mode)
        header = not os.path.exists(magpie_csv_path)
        new_row.to_csv(magpie_csv_path, mode='a', header=header, index=False)
        print(f"   [Success] Features generated and appended to {magpie_csv_path}")

        return new_row

    except Exception as e:
        print(f"   [Error] Failed to generate features: {e}")
        return None

# --- 2. DEFINE PREDICTION FUNCTION ---

def load_prediction_assets_lgbm():
    """Loads all necessary files (model, scalers, lookup table)."""
    assets = {}
    try:
        # Load the trained LGBM model
        assets['model'] = joblib.load('To_Publish_Young_Modulus_Best_On_CNN.pkl')
        
        # Load the 3 scalers (the "keys")
        assets['cnn_scaler'] = joblib.load('young_cnn_scaler_on_cnn.pkl')
        assets['pca'] = joblib.load('young_pca_on_cnn.pkl')
        assets['magpie_scaler'] = joblib.load('young_magpie_scaler_on_cnn.pkl')
        
        # Load the Magpie feature lookup table
        assets['magpie_csv_path'] = 'magpie_features.csv' # Store path
        assets['magpie_lookup_df'] = pd.read_csv(assets['magpie_csv_path'])
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        print("All LGBM model assets loaded successfully.")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Please ensure all files are in the same directory:")
        print("[To_Publish_Young_Modulus_Best_On_CNN.pkl, young_cnn_scaler_on_cnn.pkl, young_pca_on_cnn.pkl, young_magpie_scaler_on_cnn.pkl, magpie_features.csv]")
        print(f"\nDetails: {e}")
        return None
    return assets

def predict_youngs_modulus_lgbm(formula: str, npy_file_path: str, assets: dict) -> float:
    """
    Predicts the Young's Modulus (GPa) using the Tree-Fusion (LGBM) model.
    """
    if not assets:
        print("Error: Model assets are not loaded.")
        return None
        
    print(f"\n--- Predicting for {formula} ---")
    
    # === A: PROCESS 3D CNN FEATURES (PCA) ===
    try:
        x_3d_raw = np.load(npy_file_path)
    except FileNotFoundError:
        print(f"Error: Cannot find file {npy_file_path}")
        return None
        
    # 1. Flatten
    x_3d_flat = x_3d_raw.reshape(1, -1)
    # 2. Scale
    x_3d_scaled = assets['cnn_scaler'].transform(x_3d_flat)
    # 3. Apply PCA
    x_pca = assets['pca'].transform(x_3d_scaled)
    # print(f"PCA feature shape: {x_pca.shape}")

    # === B: PROCESS MAGPIE FEATURES ===
    try:
        # 1. Try to find features in loaded DataFrame
        x_1d_features = assets['magpie_lookup_df'][assets['magpie_lookup_df']['formula_sp'] == formula][assets['magpie_cols']]
        
        # 2. If not found, GENERATE THEM
        if x_1d_features.empty:
            new_row = generate_missing_magpie_features(formula, assets['magpie_csv_path'], assets['magpie_lookup_df'])
            
            if new_row is not None:
                # Use the newly generated row
                x_1d_features = new_row[assets['magpie_cols']]
                # Update memory
                assets['magpie_lookup_df'] = pd.concat([assets['magpie_lookup_df'], new_row], ignore_index=True)
            else:
                print(f"Error: Could not generate Magpie features for '{formula}'")
                return None
    except Exception as e:
        print(f"Error looking up/generating Magpie features: {e}")
        return None
    
    # 1. Scale
    x_magpie_scaled = assets['magpie_scaler'].transform(x_1d_features)
    # print(f"Magpie feature shape: {x_magpie_scaled.shape}")

    # === C: COMBINE AND PREDICT ===
    x_combined = np.concatenate([x_pca, x_magpie_scaled], axis=1)
    # print(f"Final combined feature vector shape: {x_combined.shape}")
    
    try:
        # Predict (LGBM returns GPa directly)
        final_prediction_gpa = assets['model'].predict(x_combined)[0]
        return final_prediction_gpa
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

# --- 3. EXAMPLE USAGE ---
if __name__ == "__main__":
    
    # Load all assets once
    model_assets = load_prediction_assets_lgbm()
    
    if model_assets:
        # --- Example 1: Use a sample from the Magpie CSV ---
        # (Or fallback to user input if CSV is empty/new)
        print("\n" + "="*50)
        print("Usage: python predict_young_on_cnn.py <formula_string> <path_to_npy_file>")
        
        if len(sys.argv) == 3:
            formula_from_gui = sys.argv[1]
            npy_path_from_gui = sys.argv[2]
            
            prediction_gui = predict_youngs_modulus_lgbm(
                formula=formula_from_gui,
                npy_file_path=npy_path_from_gui,
                assets=model_assets
            )
            if prediction_gui is not None:
                print(f"-> Final Prediction for {formula_from_gui} is {prediction_gui:.2f} GPa")
        else:
            # Simple test case if no args provided
            if not model_assets['magpie_lookup_df'].empty:
                test_formula = model_assets['magpie_lookup_df'].iloc[0]['formula_sp']
                test_npy_file = f"input_cnn/{test_formula}_latent.npy"
                print(f"No arguments. Testing with first row of CSV: {test_formula}")
                
                if os.path.exists(test_npy_file):
                    predict_youngs_modulus_lgbm(test_formula, test_npy_file, model_assets)
                else:
                    print(f"Skipping test: {test_npy_file} not found.")
                    
        print("="*50)