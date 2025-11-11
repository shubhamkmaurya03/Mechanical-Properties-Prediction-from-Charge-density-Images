import numpy as np
import pandas as pd
import joblib
import os
import sys
import warnings

# --- 0. SETUP ---
warnings.filterwarnings('ignore')

# --- 1. DEFINE PREDICTION FUNCTION ---

def load_prediction_assets():
    """Loads all necessary files for the 'Best' (LGBM) model."""
    assets = {}
    print("Loading models and preprocessors for 'Best' (LGBM) model...")
    try:
        assets['model'] = joblib.load('To_Publish_Shear_Modulus_Best_On_CNN.pkl')
        assets['x_flat_scaler'] = joblib.load('x_flat_scaler_shear_modulus_best.pkl')
        assets['pca_model'] = joblib.load('pca_shear_modulus_best.pkl')
        assets['magpie_scaler'] = joblib.load('magpie_scaler_shear_modulus_best.pkl')
        assets['magpie_lookup_df'] = pd.read_csv('magpie_features_shear.csv')
        assets['magpie_lookup_df'] = assets['magpie_lookup_df'].set_index('formula_sp')
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        print("All assets loaded successfully.")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Details: {e}")
        print("Please ensure all 5 files are in the same directory:")
        print("1. To_Publish_Shear_Modulus_Best.pkl")
        print("2. x_flat_scaler_shear_modulus_best.pkl")
        print("3. pca_shear_modulus_best.pkl")
        print("4. magpie_scaler_shear_modulus_best.pkl")
        print("5. magpie_features_shear.csv")
        return None
    return assets

def predict_shear_modulus(formula_sp: str, npy_file_path: str, assets: dict) -> float:
    """
    Predicts the shear modulus (in GPa) from a formula_sp and a .npy file.
    
    Args:
        formula_sp (str): The formula_sp string, e.g., "Ac2CuGe_225"
        npy_file_path (str): The path to the corresponding _latent.npy file
        assets (dict): The dictionary of loaded model assets.
    
    Returns:
        float: The predicted shear modulus in GPa, or None if an error occurs.
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
        magpie_vector_unscaled = assets['magpie_lookup_df'].loc[formula_sp][assets['magpie_cols']].values
        X_magpie_input = assets['magpie_scaler'].transform(magpie_vector_unscaled.reshape(1, -1))
    except KeyError:
        print(f"Error: Formula '{formula_sp}' not found in 'magpie_features_shear.csv'.")
        return None
    except Exception as e:
        print(f"Error processing Magpie features for '{formula_sp}': {e}")
        return None

    # === C: COMBINE FEATURES ---
    X_combined = np.concatenate([X_pca_input, X_magpie_input], axis=1)

    try:
        prediction = assets['model'].predict(X_combined)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

    return prediction[0]

# --- 3. Main execution block ---
if __name__ == "__main__":
    
    # Load all assets once
    model_assets = load_prediction_assets()
    
    if model_assets:
        # --- Example 1: Use a sample from the Magpie CSV ---
        test_formula_sp = model_assets['magpie_lookup_df'].index[0] # Get first formula_sp
        test_npy_file = f"input_cnn/{test_formula_sp}_latent.npy"
        
        print("\n" + "="*50)
        print("Running Example 1...")
        
        if not os.path.exists(test_npy_file):
            print(f"Warning: Example file {test_npy_file} not found. Skipping example 1.")
        else:
            prediction = predict_shear_modulus(
                formula_sp=test_formula_sp,
                npy_file_path=test_npy_file,
                assets=model_assets
            )
            if prediction is not None:
                print(f"-> Example 1 Prediction for {test_formula_sp} is {prediction:.2f} GPa")
        
        # --- Example 2: Handle user input ---
        print("\n" + "="*50)
        print("Running Example 2 (using command-line arguments)...")
        
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
            print("To run a custom prediction, use the format:")
            print("python predict_shear_modulus_best.py <formula_sp> <path_to_npy_file>")
            
        print("="*50)