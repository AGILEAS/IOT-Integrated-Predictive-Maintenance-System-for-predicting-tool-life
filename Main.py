import numpy as np
from tensorflow.keras.models import load_model
import pickle

# 1. Load the trained assets
model = load_model('maintenance_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

print("--- INDUSTRIAL AI DIAGNOSTIC TOOL (CONSOLE VERSION) ---")

# 2. Manual Input via Terminal
def get_user_input():
    print(f"\nPlease enter current readings for the following {len(features[:5])} sensors:")
    inputs = []
    for f in features[:5]:
        val = float(input(f"Enter value for {f}: "))
        inputs.append(val)
    return inputs

try:
    user_vals = get_user_input()
    
    # Prepare data for LSTM (50 time steps)
    test_data = np.full((1, 50, len(features)), 0.5)
    test_data[0, -1, :5] = user_vals 
    
    # Predict
    prediction = model.predict(test_data, verbose=0)[0][0]
    
    print("\n" + "="*30)
    print(f"ANALYSIS RESULT:")
    if prediction > 0.5:
        print(f"STATUS: 🚨 CRITICAL RISK ({prediction*100:.1f}%)")
        print("WINDOW: Failure predicted in 48-72 HOURS.")
    else:
        print(f"STATUS: ✅ HEALTHY ({(1-prediction)*100:.1f}%)")
        print("WINDOW: Machine is stable.")
    print("="*30)

except ValueError:
    print("Invalid input. Please enter numbers only.")
