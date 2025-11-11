import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import sys
import warnings

# --- Import Custom Layers ---
try:
    from custom_layers_shear_modulus_cnn import DualAttention3D, RandomFlip3D
except ImportError:
    print("Error: Could not import custom_layers_shear_modulus_cnn.py.")
    print("Please ensure 'custom_layers_shear_modulus_cnn.py' is in the same directory.")
    exit()

# --- 0. SETUP ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- 1. DEFINE PREDICTION FUNCTION ---

def load_prediction_assets():
    """Loads all necessary files for the 3-Branch Keras Fusion Model."""
    assets = {}
    print("Loading models and preprocessors for 3-Branch CNN Model...")
    try:
        # Load the trained Keras model
        assets['model'] = load_model(
            'To_Publish_Shear_Modulus_Best.h5', 
            custom_objects={
                'DualAttention3D': DualAttention3D,
                'RandomFlip3D': RandomFlip3D
            }
        )
        
        # Load the 5 scalers/processors
        assets['cnn_max_val'] = joblib.load('cnn_max_value_shear_modulus_cnn.pkl')
        assets['magpie_scaler'] = joblib.load('magpie_scaler_shear_modulus_cnn.pkl')
        assets['y_scaler'] = joblib.load('y_scaler_shear_modulus_cnn.pkl')
        assets['x_flat_scaler'] = joblib.load('x_flat_scaler_shear_modulus_cnn.pkl')
        assets['pca_model'] = joblib.load('pca_shear_modulus_cnn.pkl')
        
        # Load the Magpie feature lookup table
        assets['magpie_lookup_df'] = pd.read_csv('magpie_features_shear.csv')
        assets['magpie_lookup_df'] = assets['magpie_lookup_df'].set_index('formula_sp')
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        print("All assets loaded successfully.")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Details: {e}")
        print("Please ensure all 8 files are in the same directory:")
        print("1. To_Publish_Shear_Modulus_Best.h5")
        print("2. custom_layers_shear_modulus_cnn.py")
        print("3. cnn_max_value_shear_modulus_cnn.pkl")
        print("4. magpie_scaler_shear_modulus_cnn.pkl")
        print("5. y_scaler_shear_modulus_cnn.pkl")
        print("6. x_flat_scaler_shear_modulus_cnn.pkl")
        print("7. pca_shear_modulus_cnn.pkl")
        print("8. magpie_features_shear.csv")
        return None
    return assets

def predict_shear_modulus(formula_sp: str, npy_file_path: str, assets: dict) -> float:
    """
    Predicts the shear modulus (GPa) from a formula_sp and a .npy file.
    
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
    
    # === A: PROCESS 3D CNN INPUT (Input 1) ===
    try:
        X_3d_raw = np.load(npy_file_path)
    except FileNotFoundError:
        print(f"Error: Cannot find file {npy_file_path}")
        return None
    except Exception as e:
        print(f"Error processing .npy file '{npy_file_path}': {e}")
        return None

    X_3d_scaled = X_3d_raw / assets['cnn_max_val']
    X_3d_input = np.expand_dims(X_3d_scaled, axis=0)

    # === B: PROCESS PCA INPUT (Input 2) ===
    try:
        X_3d_flat = X_3d_raw.reshape(1, -1)
        X_3d_flat_scaled = assets['x_flat_scaler'].transform(X_3d_flat)
        X_pca_input = assets['pca_model'].transform(X_3d_flat_scaled)
    except Exception as e:
        print(f"Error processing PCA features: {e}")
        return None

    # === C: PROCESS MAGPIE INPUT (Input 3) ===
    try:
        magpie_vector_unscaled = assets['magpie_lookup_df'].loc[formula_sp][assets['magpie_cols']].values
        X_magpie_input = assets['magpie_scaler'].transform(magpie_vector_unscaled.reshape(1, -1))
    except KeyError:
        print(f"Error: Formula '{formula_sp}' not found in 'magpie_features_shear.csv'.")
        return None
    except Exception as e:
        print(f"Error processing Magpie features for '{formula_sp}': {e}")
        return None

    # === D: MAKE PREDICTION ===
    # The model expects a dictionary with these *exact* keys
    model_inputs = {
        "input_3d": X_3d_input, 
        "input_pca": X_pca_input,
        "input_magpie": X_magpie_input
    }
    
    try:
        y_pred_scaled = assets['model'].predict(model_inputs, verbose=0)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

    # === E: Inverse Transform (Un-scale) ---
    y_pred_real = assets['y_scaler'].inverse_transform(y_pred_scaled)
    
    return y_pred_real[0][0] # Return the single GPa value

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
            print("python predict_shear_modulus_cnn.py <formula_sp> <path_to_npy_file>")
            
        print("="*50)