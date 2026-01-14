
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

def refresh_model(model_path):
    print(f"Processing {model_path}...")
    try:
        # Load with compile=False to avoid optimizer issues
        model = load_model(model_path, compile=False)
        
        # Create a dummy input to trace/build the model if needed (optional but good sanity check)
        # Malaria: 50x50x3, Pneumonia: 64x64x3
        if 'model111' in model_path:
            input_shape = (1, 50, 50, 3)
        else:
            input_shape = (1, 64, 64, 3)
            
        dummy_input = np.zeros(input_shape)
        pred = model.predict(dummy_input)
        print(f"Prediction test successful. Output shape: {pred.shape}")
        
        # Context manager to ensure file handle is closed before writing? 
        # Usually save handles it. 
        # We will save to a temporary name first then rename
        new_path = model_path # Overwrite
        
        # Save without optimizer to reduce size and errors (we only need inference)
        model.save(new_path, include_optimizer=False)
        print(f"Successfully re-saved {model_path}")
        
    except Exception as e:
        print(f"Error processing {model_path}: {e}")

if __name__ == "__main__":
    refresh_model('model111.h5') # Malaria
    refresh_model('my_model.h5') # Pneumonia
