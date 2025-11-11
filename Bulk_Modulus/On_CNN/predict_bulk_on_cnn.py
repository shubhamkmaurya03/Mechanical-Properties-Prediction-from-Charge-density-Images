import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
import sys
import warnings

# --- 0. SETUP ---
warnings.filterwarnings('ignore')

# --- 1. DEFINE PREDICTION FUNCTION ---

def load_prediction_assets_on_cnn():
    """Loads all necessary files (model, scalers, lookup table)."""
    assets = {}
    try:
        assets['model'] = joblib.load('To_Publish_Bulk_Modulus_LGBM.pkl')
        assets['cnn_scaler'] = joblib.load('bulk_pca_scaler_on_cnn.pkl')
        assets['pca'] = joblib.load('bulk_pca_model_on_cnn.pkl')
        assets['magpie_scaler'] = joblib.load('bulk_magpie_scaler_on_cnn.pkl')
        assets['magpie_lookup_df'] = pd.read_csv('magpie_features.csv')
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        print("All Bulk Modulus (LGBM / on_cnn) assets loaded successfully.")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Please ensure all files are in the same directory:")
        print("[To_Publish_Bulk_Modulus_Best_On_CNN.pkl, bulk_pca_scaler_on_cnn.pkl, bulk_pca_model_on_cnn.pkl, bulk_magpie_scaler_on_cnn.pkl, magpie_features.csv]")
        print(f"\nDetails: {e}")
        return None
    return assets

def predict_bulk_modulus_on_cnn(formula: str, npy_file_path: str, assets: dict) -> float:
    """
    Predicts the Bulk Modulus (GPa) using the Tree-Fusion (LGBM) model.
    
    Args:
        formula (str): The formula string, e.g., "Ac2CuGe_225"
        npy_file_path (str): The path to the corresponding _latent.npy file
        assets (dict): The dictionary of loaded model assets.
    
    Returns:
        float: The predicted Bulk Modulus in GPa, or None if an error occurs.
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
       
    x_3d_flat = x_3d_raw.reshape(1, -1)
    x_3d_scaled = assets['cnn_scaler'].transform(x_3d_flat)
    x_pca = assets['pca'].transform(x_3d_scaled)
    print(f"PCA feature shape: {x_pca.shape}")

    # === B: PROCESS MAGPIE FEATURES ===
    try:
        x_1d_features = assets['magpie_lookup_df'][assets['magpie_lookup_df']['formula_sp'] == formula][assets['magpie_cols']]
        if x_1d_features.empty:
            print(f"Error: Formula '{formula}' not found in magpie_features.csv")
            return None
    except Exception as e:
        print(f"Error looking up Magpie features: {e}")
        return None
    
    x_magpie_scaled = assets['magpie_scaler'].transform(x_1d_features)
    print(f"Magpie feature shape: {x_magpie_scaled.shape}")

    # === C: COMBINE AND PREDICT ===
    x_combined = np.concatenate([x_pca, x_magpie_scaled], axis=1)
    print(f"Final combined feature vector shape: {x_combined.shape}")
    
    try:
        final_prediction_gpa = assets['model'].predict(x_combined)[0]
        return final_prediction_gpa
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

# --- 3. EXAMPLE USAGE (for your supervisor to test) ---
if __name__ == "__main__":
    
    model_assets = load_prediction_assets_on_cnn()
    
    if model_assets:
        # --- Example 1: Use a sample from the Magpie CSV ---
        test_formula = model_assets['magpie_lookup_df'].iloc[0]['formula_sp']
        test_npy_file = f"input_cnn/{test_formula}_latent.npy"
        
        print("\n" + "="*50)
        print("Running Example 1...")
        
        if not os.path.exists(test_npy_file):
            print(f"Warning: Example file {test_npy_file} not found. Skipping example 1.")
        else:
            prediction = predict_bulk_modulus_on_cnn(
                formula=test_formula,
                npy_file_path=test_npy_file,
                assets=model_assets
            )
            if prediction is not None:
                print(f"-> Example 1 Prediction for {test_formula} is {prediction:.2f} GPa")
        
        # --- Example 2: Handle user input ---
        print("\n" + "="*50)
        print("Running Example 2 (using command-line arguments)...")
        
        if len(sys.argv) == 3:
            formula_from_gui = sys.argv[1]
            npy_path_from_gui = sys.argv[2]
            
            prediction_gui = predict_bulk_modulus_on_cnn(
                formula=formula_from_gui,
                npy_file_path=npy_path_from_gui,
                assets=model_assets
            )
            if prediction_gui is not None:
                print(f"-> Final Prediction for {formula_from_gui} is {prediction_gui:.2f} GPa")
        else:
            print("To run a custom prediction, use the format:")
            print("python predict_bulk_on_cnn.py <formula_string> <path_to_npy_file>")
            
        print("="*50)