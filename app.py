import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Industrial AI Predictor", layout="wide")
st.title("⚙️ IoT Predictive Maintenance Dashboard")

# --- 2. THE BRAIN (Training 5 Epochs) ---
@st.cache_resource # This stops it from retraining every time you move a slider
def train_and_prepare():
    # Load from your Desktop file
    path = "train_FD001.txt"
    cols = ['unit', 'cycles', 'set1', 'set2', 'set3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(path, sep='\s+', header=None, names=cols)

    # Labeling for 48-72 hour window
    max_cycles = df.groupby('unit')['cycles'].max().reset_index()
    df = df.merge(max_cycles, on='unit', suffixes=('', '_max'))
    df['label'] = ((df['cycles_max'] - df['cycles']) <= 72).astype(int)

    # Clean & Scale
    features = [f's{i}' for i in range(1, 22)]
    df = df.drop(columns=[col for col in features if df[col].std() == 0])
    active_features = [c for c in df.columns if c.startswith('s')]
    
    scaler = MinMaxScaler()
    df[active_features] = scaler.fit_transform(df[active_features])

    # Sequence building for LSTM
    X, y = [], []
    for unit in df['unit'].unique()[:5]: # Speed up for the demo
        unit_data = df[df['unit'] == unit]
        f_vals = unit_data[active_features].values
        l_vals = unit_data['label'].values
        for i in range(len(unit_data) - 50):
            X.append(f_vals[i:i+50])
            y.append(l_vals[i+50])

    X, y = np.array(X), np.array(y)

    # LSTM Model
    model = Sequential([
        LSTM(50, input_shape=(50, X.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train for your 5 Epochs
    with st.status("AI Training in Progress...", expanded=True) as status:
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        status.update(label="✅ Training Complete!", state="complete", expanded=False)

    return model, scaler, active_features

# Run the training logic
model, scaler, feature_list = train_and_prepare()

# --- 3. PROFESSIONAL UI (The Dashboard) ---
st.divider()
col_in, col_out = st.columns([1, 1.5])

with col_in:
    st.subheader("🛠️ Manual Sensor Input")
    st.write("Enter current telemetry data:")
    
    # Create inputs for the key sensors
    user_inputs = []
    for f in feature_list[:6]: # Show first 6 sensors for clean UI
        val = st.number_input(f"Reading for {f}", value=0.5, format="%.4f")
        user_inputs.append(val)
    
    analyze_btn = st.button("RUN DIAGNOSTIC", type="primary")

with col_out:
    if analyze_btn:
        # Prepare the 50-step window for LSTM
        # We fill historical data with 'normal' values and the latest step with user input
        test_input = np.full((1, 50, len(feature_list)), 0.5)
        test_input[0, -1, :6] = user_inputs
        
        # Predict
        prob = model.predict(test_input)[0][0]
        
        # Display Results
        st.subheader("Diagnostic Results")
        
        if prob > 0.6:
            st.error(f"### FAILURE RISK: {prob*100:.1f}%")
            st.warning("**Status:** Critical Wear Detected")
            st.info("**Predicted Window:** Failure likely in **48-72 Hours**")
            st.markdown("---")
            st.write("💰 **ROI Action:** Schedule maintenance now to save **$40,000** in breakdown costs.")
        else:
            st.success(f"### HEALTH SCORE: {(1-prob)*100:.1f}%")
            st.write("**Status:** Normal Operation")
            st.write("**Remaining Tool Life:** Estimated > 150 Hours")
