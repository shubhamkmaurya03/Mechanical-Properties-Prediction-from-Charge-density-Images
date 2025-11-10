import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
import joblib
import warnings
from custom_layers_young import DualAttention3D
import os
import sys

# --- 0. SETUP ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. DEFINE PREDICTION FUNCTION ---

def load_prediction_assets():
    """Loads all necessary files (model, scalers, lookup table)."""
    assets = {}
    try:
        # Load the custom layer
        tf.keras.utils.get_custom_objects()['DualAttention3D'] = DualAttention3D
        
        # Load the trained model
        assets['model'] = models.load_model('To_Publish_Young_Modulus_Best.h5')
        
        # Load the scalers (the "keys")
        assets['cnn_max_val'] = joblib.load('cnn_max_value_young.pkl')
        assets['magpie_scaler'] = joblib.load('magpie_scaler_young.pkl')
        assets['y_scaler'] = joblib.load('y_scaler_young.pkl')
        
        # Load the Magpie feature lookup table
        assets['magpie_lookup_df'] = pd.read_csv('magpie_features.csv')
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        print("All model assets loaded successfully.")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Please ensure all files are in the same directory:")
        print("[To_Publish_Young_Modulus_Best.h5, custom_layers.py, cnn_max_value.pkl, magpie_scaler.pkl, y_scaler.pkl, magpie_features.csv]")
        print(f"\nDetails: {e}")
        return None
    return assets

def predict_youngs_modulus(formula: str, npy_file_path: str, assets: dict) -> float:
    """
    Predicts the Young's Modulus (GPa) from a formula and a .npy file.
    
    Args:
        formula (str): The formula string, e.g., "Ac2CuGe_225"
        npy_file_path (str): The path to the corresponding _latent.npy file
        assets (dict): The dictionary of loaded model assets.
    
    Returns:
        float: The predicted Young's Modulus in GPa, or None if an error occurs.
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
    x_3d_final = np.expand_dims(x_3d_scaled, axis=0)

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

    # === C: MAKE PREDICTION ===
    # The model expects a dictionary with these *exact* keys
    model_inputs = {
        "input_layer": x_3d_final, 
        "Magpie_Input": x_1d_scaled
    }
    
    try:
        y_pred_scaled = assets['model'].predict(model_inputs, verbose=0)
        y_pred_real = assets['y_scaler'].inverse_transform(y_pred_scaled)
        final_prediction_gpa = y_pred_real[0][0]
        return final_prediction_gpa
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

# --- 3. EXAMPLE USAGE (for your supervisor to test) ---
if __name__ == "__main__":
    
    # Load all assets once
    model_assets = load_prediction_assets()
    
    if model_assets:
        # --- Example 1: Use a sample from the Magpie CSV ---
        test_formula = model_assets['magpie_lookup_df'].iloc[0]['formula_sp']
        test_npy_file = f"input_cnn/{test_formula}_latent.npy"
        
        print("\n" + "="*50)
        print("Running Example 1...")
        
        if not os.path.exists(test_npy_file):
            print(f"Warning: Example file {test_npy_file} not found. Skipping example 1.")
        else:
            prediction = predict_youngs_modulus(
                formula=test_formula,
                npy_file_path=test_npy_file,
                assets=model_assets
            )
            if prediction is not None:
                print(f"-> Example 1 Prediction for {test_formula} is {prediction:.2f} GPa")
        
        # --- Example 2: Handle user input ---
        print("\n" + "="*50)
        print("Running Example 2 (using command-line arguments)...")
        
        # This allows the GUI to call: python predict.py "MyFormula_123" "path/to/file.npy"
        if len(sys.argv) == 3:
            formula_from_gui = sys.argv[1]
            npy_path_from_gui = sys.argv[2]
            
            prediction_gui = predict_youngs_modulus(
                formula=formula_from_gui,
                npy_file_path=npy_path_from_gui,
                assets=model_assets
            )
            if prediction_gui is not None:
                print(f"-> Final Prediction for {formula_from_gui} is {prediction_gui:.2f} GPa")
        else:
            print("To run a custom prediction, use the format:")
            print("python predict.py <formula_string> <path_to_npy_file>")
            
        print("="*50)