import numpy as np
import pandas as pd
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
        # Note: assets_df index is formula_sp, so we look at columns directly
        required_cols = [col for col in assets_df.columns]
        
        # Check if generation failed (sometimes matminer returns NaNs for invalid elements)
        if df_features[required_cols].isnull().values.any():
            print(f"   [Error] Generated features contain NaNs for {formula_clean}.")
            return None

        # Prepare final row (formula_sp + features)
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

def load_prediction_assets():
    """Loads all necessary files for the 'Best' (LGBM) model."""
    assets = {}
    print("Loading models and preprocessors for 'Best' (LGBM) model...")
    try:
        assets['model'] = joblib.load('To_Publish_Shear_Modulus_Best_On_CNN.pkl')
        assets['x_flat_scaler'] = joblib.load('x_flat_scaler_shear_modulus_best.pkl')
        assets['pca_model'] = joblib.load('pca_shear_modulus_best.pkl')
        assets['magpie_scaler'] = joblib.load('magpie_scaler_shear_modulus_best.pkl')
        
        # Load the Magpie feature lookup table
        assets['magpie_csv_path'] = 'magpie_features_shear.csv' # Store path
        assets['magpie_lookup_df'] = pd.read_csv(assets['magpie_csv_path'])
        
        # Store columns list before setting index
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        # Set index for faster lookup, but keep a copy or logic to handle updates
        assets['magpie_lookup_df'] = assets['magpie_lookup_df'].set_index('formula_sp')
        
        print("All assets loaded successfully.")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Details: {e}")
        print("Please ensure all 5 files are in the same directory:")
        print("1. To_Publish_Shear_Modulus_Best_On_CNN.pkl")
        print("2. x_flat_scaler_shear_modulus_best.pkl")
        print("3. pca_shear_modulus_best.pkl")
        print("4. magpie_scaler_shear_modulus_best.pkl")
        print("5. magpie_features_shear.csv")
        return None
    return assets

def predict_shear_modulus(formula_sp: str, npy_file_path: str, assets: dict) -> float:
    """
    Predicts the shear modulus (in GPa) from a formula_sp and a .npy file.
    """
    if not assets:
        print("Error: Model assets are not loaded.")
        return None
        
    print(f"\n--- Predicting for {formula_sp} ---")
    
    # === A: PROCESS 3D-to-PCA INPUT (Input 1) ===
    try:
        X_3d_raw = np.load(npy_file_path)
    except FileNotFoundError:
        print(f"Error: Cannot find file {npy_file_path}")
        return None
    except Exception as e:
        print(f"Error processing .npy file '{npy_file_path}': {e}")
        return None

    try:
        X_3d_flat = X_3d_raw.reshape(1, -1)
        X_3d_flat_scaled = assets['x_flat_scaler'].transform(X_3d_flat)
        X_pca_input = assets['pca_model'].transform(X_3d_flat_scaled) # Shape: (1, 512)
    except Exception as e:
        print(f"Error processing PCA features: {e}")
        return None

    # === B: PROCESS MAGPIE INPUT (Input 2) ===
    try:
        # 1. Try to find features in loaded DataFrame (using index)
        if formula_sp in assets['magpie_lookup_df'].index:
            magpie_vector_unscaled = assets['magpie_lookup_df'].loc[formula_sp][assets['magpie_cols']].values
            magpie_vector_unscaled = magpie_vector_unscaled.reshape(1, -1)
        else:
            # 2. If not found, GENERATE THEM
            new_row = generate_missing_magpie_features(formula_sp, assets['magpie_csv_path'], assets['magpie_lookup_df'])
            
            if new_row is not None:
                # Extract features for prediction
                magpie_vector_unscaled = new_row[assets['magpie_cols']].values
                
                # Update memory (reset index to append, then set index back)
                # A bit clunky but safe for pandas indexing
                temp_df = assets['magpie_lookup_df'].reset_index()
                temp_df = pd.concat([temp_df, new_row], ignore_index=True)
                assets['magpie_lookup_df'] = temp_df.set_index('formula_sp')
            else:
                print(f"Error: Could not generate Magpie features for '{formula_sp}'")
                return None
                
        X_magpie_input = assets['magpie_scaler'].transform(magpie_vector_unscaled)
        
    except Exception as e:
        print(f"Error processing Magpie features for '{formula_sp}': {e}")
        return None

    # === C: COMBINE FEATURES ---
    X_combined = np.concatenate([X_pca_input, X_magpie_input], axis=1)

    try:
        prediction = assets['model'].predict(X_combined)
        return prediction[0]
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

# --- 3. Main execution block ---
if __name__ == "__main__":
    
    # Load all assets once
    model_assets = load_prediction_assets()
    
    if model_assets:
        # --- Example 1: Use a sample from the Magpie CSV ---
        # (Or fallback to user input if CSV is empty/new)
        print("\n" + "="*50)
        print("Usage: python predict_shear_modulus_best.py <formula_sp> <path_to_npy_file>")
        
        # Check for command line arguments
        if len(sys.argv) == 3:
            formula_from_gui = sys.argv[1]
            npy_path_from_gui = sys.argv[2]
            
            prediction_gui = predict_shear_modulus(
                formula_sp=formula_from_gui,
                npy_file_path=npy_path_from_gui,
                assets=model_assets
            )
            if prediction_gui is not None:
                print(f"-> Final Prediction for {formula_from_gui} is {prediction_gui:.2f} GPa")
        else:
            # Simple test case if no args provided
            if not model_assets['magpie_lookup_df'].empty:
                test_formula_sp = model_assets['magpie_lookup_df'].index[0] # Get first formula_sp
                test_npy_file = f"input_cnn/{test_formula_sp}_latent.npy"
                print(f"No arguments. Testing with first row of CSV: {test_formula_sp}")
                
                if os.path.exists(test_npy_file):
                    predict_shear_modulus(test_formula_sp, test_npy_file, model_assets)
                else:
                    print(f"Skipping test: {test_npy_file} not found.")

        print("="*50)