import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import pandas as pd
import joblib
import warnings
import os
import sys

# --- 0. SETUP ---
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow warnings

# CRITICAL: Import the custom layer definition
try:
    from custom_layers_fe import DualAttention3D
except ImportError:
    print("--- FATAL ERROR: 'custom_layers.py' not found. ---")
    print("Please make sure 'custom_layers.py' is in the same directory.")
    sys.exit(1)

# --- 1. DEFINE PREDICTION FUNCTIONS ---

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
        assets['magpie_lookup_df'] = pd.read_csv('magpie_features.csv')
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        print("All 3D-CNN model assets loaded successfully.")
        
    except FileNotFoundError as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Missing file: {e.filename}")
        print("Please ensure all files are in the same directory:")
        print("[To_Publish_Formation_Energy_Best_CNN.h5, custom_layers.py, cnn_max_value.pkl, magpie_scaler.pkl, y_scaler.pkl, magpie_features.csv]")
        return None
    except Exception as e:
        print(f"An unknown error occurred loading assets: {e}")
        return None
    return assets

def predict_formation_energy_cnn(formula: str, npy_file_path: str, assets: dict) -> float:
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

# --- 2. EXAMPLE USAGE ---
if __name__ == "__main__":
    
    # Load all assets once at the start
    model_assets = load_prediction_assets_cnn()
    
    if model_assets:
        # --- Example 1: Use a sample from the Magpie CSV ---
        test_formula = model_assets['magpie_lookup_df'].iloc[0]['formula_sp']
        test_npy_file = f"input_cnn/{test_formula}_latent.npy"
        
        print("\n" + "="*50)
        print("Running Example 1...")
        
        if not os.path.exists(test_npy_file):
            print(f"Warning: Example file {test_npy_file} not found. Skipping example 1.")
        else:
            prediction = predict_formation_energy_cnn(
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
            
            prediction_gui = predict_formation_energy_cnn(
                formula=formula_from_gui,
                npy_file_path=npy_path_from_gui,
                assets=model_assets
            )
            if prediction_gui is not None:
                print(f"-> Final Prediction for {formula_from_gui} is {prediction_gui:.4f} eV/atom")
        else:
            print("To run a custom prediction, use the format:")
            print("python predict_formation_energy_cnn.py <formula_string> <path_to_npy_file>")
            
        print("="*50)