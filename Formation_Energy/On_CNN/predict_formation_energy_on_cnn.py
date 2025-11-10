import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
import warnings
import os
import sys

# --- 0. SETUP ---
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. DEFINE PREDICTION FUNCTIONS ---

def load_prediction_assets_lgbm():
    """Loads all necessary files (model, scalers, PCA, lookup table)."""
    assets = {}
    try:
        # Load the trained model
        assets['model'] = joblib.load('To_Publish_Formation_Energy_Best_On_CNN.pkl')
        
        # Load the preprocessors (the "keys")
        assets['scaler_3d'] = joblib.load('scaler_3d_fe_on_cnn.pkl')
        assets['pca'] = joblib.load('pca_transformer_fe_on_cnn.pkl')
        assets['magpie_scaler'] = joblib.load('magpie_scaler_fe_on_cnn.pkl')
        
        # Load the Magpie feature lookup table
        assets['magpie_lookup_df'] = pd.read_csv('magpie_features.csv')
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        print("All LGBM model assets loaded successfully.")
        
    except FileNotFoundError as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Missing file: {e.filename}")
        print("Please ensure all files are in the same directory:")
        print("[To_Publish_Formation_Energy_Best_On_CNN.pkl, scaler_3d.pkl, pca_transformer.pkl, magpie_scaler.pkl, magpie_features.csv]")
        return None
    except Exception as e:
        print(f"An unknown error occurred loading assets: {e}")
        return None
    return assets

def predict_formation_energy_lgbm(formula: str, npy_file_path: str, assets: dict) -> float:
    """
    Predicts the Formation Energy (eV/atom) from a formula and a .npy file.
    
    Args:
        formula (str): The formula string, e.g., "Ag2O_225"
        npy_file_path (str): The path to the corresponding _latent.npy file
        assets (dict): The dictionary of loaded model assets.
    
    Returns:
        float: The predicted Formation Energy in eV/atom, or None if an error occurs.
    """
    if not assets:
        print("Error: Model assets are not loaded.")
        return None
        
    print(f"\n--- Predicting for {formula} ---")
    
    # === A: PROCESS 3D (PCA) INPUT ===
    try:
        x_3d = np.load(npy_file_path)
    except FileNotFoundError:
        print(f"Error: Cannot find file {npy_file_path}")
        return None
        
    x_3d_flat = x_3d.reshape(1, -1)
    x_3d_scaled = assets['scaler_3d'].transform(x_3d_flat)
    x_pca = assets['pca'].transform(x_3d_scaled)

    # === B: PROCESS MAGPIE INPUT ===
    try:
        x_1d_features = assets['magpie_lookup_df'][assets['magpie_lookup_df']['formula_sp'] == formula][assets['magpie_cols']]
        if x_1d_features.empty:
            print(f"Error: Formula '{formula}' not found in magpie_features.csv")
            return None
    except Exception as e:
        print(f"Error looking up Magpie features: {e}")
        return None
    
    x_1d_scaled = assets['magpie_scaler'].transform(x_1d_features)

    # === C: COMBINE AND PREDICT ===
    try:
        X_combined = np.concatenate([x_pca, x_1d_scaled], axis=1)
        y_pred_real = assets['model'].predict(X_combined)
        
        # This model predicts the real value directly, no y_scaler needed
        final_prediction = y_pred_real[0]
        return final_prediction
        
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

# --- 2. EXAMPLE USAGE ---
if __name__ == "__main__":
    
    # Load all assets once at the start
    model_assets = load_prediction_assets_lgbm()
    
    if model_assets:
        # --- Example 1: Use a sample from the Magpie CSV ---
        test_formula = model_assets['magpie_lookup_df'].iloc[0]['formula_sp']
        test_npy_file = f"input_cnn/{test_formula}_latent.npy"
        
        print("\n" + "="*50)
        print("Running Example 1...")
        
        if not os.path.exists(test_npy_file):
            print(f"Warning: Example file {test_npy_file} not found. Skipping example 1.")
        else:
            prediction = predict_formation_energy_lgbm(
                formula=test_formula,
                npy_file_path=test_npy_file,
                assets=model_assets
            )
            if prediction is not None:
                print(f"-> Example 1 Prediction for {test_formula} is {prediction:.4f} eV/atom")
        
        # --- Example 2: Handle user input (for GUI) ---
        print("\n" + "="*50)
        print("Running Example 2 (using command-line arguments)...")
        
        if len(sys.argv) == 3:
            formula_from_gui = sys.argv[1]
            npy_path_from_gui = sys.argv[2]
            
            prediction_gui = predict_formation_energy_lgbm(
                formula=formula_from_gui,
                npy_file_path=npy_path_from_gui,
                assets=model_assets
            )
            if prediction_gui is not None:
                print(f"-> Final Prediction for {formula_from_gui} is {prediction_gui:.4f} eV/atom")
        else:
            print("To run a custom prediction, use the format:")
            print("python predict_formation_energy_on_cnn.py <formula_string> <path_to_npy_file>")
            
        print("="*50)