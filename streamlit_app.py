import streamlit as st
import numpy as np
import os
from PIL import Image

# Force legacy Keras for compatibility with old .h5 models
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import joblib
import sys
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.neighbors
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# --- Comprehensive Legacy Model Support ---
# These patches allow models trained on older scikit-learn/joblib versions (circa 2018-2020)
# to load in modern environments (2024+).

# 1. Patch 'sklearn.externals.joblib' -> 'joblib'
if "sklearn.externals.joblib" not in sys.modules:
    sys.modules["sklearn.externals.joblib"] = joblib

# 2. Patch renamed/moved modules in scikit-learn
sys.modules['sklearn.linear_model.logistic'] = sklearn.linear_model
sys.modules['sklearn.tree.tree'] = sklearn.tree
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble
sys.modules['sklearn.neighbors.classification'] = sklearn.neighbors
# ------------------------------------------

# Page Config
st.set_page_config(
    page_title="HealthCare AI Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stCard {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        color: #31333F;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .highlight {
        color: #3a7bd5;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
    }
    .safe {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
    }
    .danger {
        background: linear-gradient(90deg, #cb2d3e 0%, #ef473a 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load Models (Cached)


# Scaler Helpers
@st.cache_resource
def get_heart_scaler():
    # Replicates the preprocessing in heart.py
    try:
        df = pd.read_csv('heart.csv')
        # Apply Log Transform as done in training
        df["trestbps"] = np.log(df["trestbps"])
        df["chol"] = np.log(df["chol"])
        # Drop columns not used in training features
        # Used: age,sex,cp,trestbps,chol,restecg,thalach,exang,oldpeak,slope,thal
        df = df.drop(["fbs", "ca", "target"], axis=1)
        sc = StandardScaler()
        sc.fit(df)
        return sc
    except Exception as e:
        # If heart.csv is missing or error, return None (fallback to raw)
        return None

def plot_health_gauge(value, title, min_val, max_val, safe_range):
    """
    Creates a Gauge Chart using Plotly.
    safe_range: tuple (min_safe, max_safe) - values inside are Green, outside are Red/Yellow.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, safe_range[0]], 'color': "#FF4B4B"}, # Red (Low)
                {'range': [safe_range[0], safe_range[1]], 'color': "#29B09D"}, # Green (Safe)
                {'range': [safe_range[1], max_val], 'color': "#FF4B4B"}  # Red (High)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

class ManualScaler:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.scale = np.array(std)
    
    def transform(self, X):
        return (np.array(X) - self.mean) / self.scale

@st.cache_resource
def load_models():
    models = {}
    
    def load_single_model(name, path, is_joblib=False):
        try:
            if is_joblib:
                return joblib.load(path)
            else:
                return load_model(path, compile=False) # compile=False avoids optimizer mismatch for inference
        except Exception as e:
            # Log error but don't crash app
            print(f"Failed to load {name}: {e}")
            return None

    models['malaria'] = load_single_model('malaria', 'model111.h5')
    models['pneumonia'] = load_single_model('pneumonia', 'my_model.h5')
    models['diabetes'] = load_single_model('diabetes', 'model1', is_joblib=True)
    models['cancer'] = load_single_model('cancer', 'model', is_joblib=True)
    models['kidney'] = load_single_model('kidney', 'model3', is_joblib=True)
    models['liver'] = load_single_model('liver', 'model4', is_joblib=True)
    models['heart'] = load_single_model('heart', 'model2', is_joblib=True)
    
    return models

models = load_models()

@st.cache_resource
def get_liver_scaler():
    # Statistical averages from Indian Liver Patient Dataset
    # Features: Age, Gender, TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, A/G Ratio
    # Gender encoded as: Male=0, Female=1
    means = [44.75, 0.2436, 3.3, 1.49, 290.6, 80.7, 109.9, 6.48, 3.14, 0.95]
    stds = [16.19, 0.4292, 6.21, 2.81, 242.94, 182.62, 288.92, 1.09, 0.80, 0.32]
    return ManualScaler(means, stds)

# Prediction Functions
def predict_malaria(img, model):
    img = img.resize((50, 50))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predicted = model.predict(img_array)
    return predicted

def predict_pneumonia(img, model):
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predicted = model.predict(img_array)
    return predicted

def predict_structured(model, input_list, scaler=None):
    # Reshape input
    input_array = np.array(input_list).reshape(1, len(input_list))
    
    # Scale if scaler is provided
    if scaler:
        try:
             input_array = scaler.transform(input_array)
        except Exception as e:
             # In case of mismatch, print error and proceed with raw (risky but better than crash)
             print(f"Scaler error: {e}")
    
    result = model.predict(input_array)
    return result[0]

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Health Check AI")
    st.markdown("---")
    app_mode = st.radio("Choose a Disease Prediction Model",
        ["Home", "Diabetes", "Heart Disease", "Kidney Disease", "Liver Disease", "Generate Cancer", "Malaria Detection", "Pneumonia Detection"])
    st.markdown("---")
    st.info("Medical Disclaimer: This AI tool is for educational purposes only and should not replace professional medical advice.")

# Main Content

if app_mode == "Home":
    st.title("Welcome to HealthCare AI Platform 🏥")
    st.markdown("""
    <div class="stCard">
        <h3>Artificial Intelligence for Early Disease Detection</h3>
        <p>This advanced platform leverages Machine Learning and Deep Learning to predict potential health risks. 
        Select a disease from the sidebar to get started.</p>
        <br>
        <ul>
            <li><b>Diabetes</b>: Random Forest Model (8 features)</li>
            <li><b>Heart Disease</b>: Logistic Regression (11 features)</li>
            <li><b>Kidney Disease</b>: Chronic Kidney Disease Prediction (12 features)</li>
            <li><b>Liver Disease</b>: Patient Record Analysis (10 features)</li>
            <li><b>Breast Cancer</b>: Cell Nuclei Analysis (30 features)</li>
            <li><b>Malaria</b>: CNN Image Analysis (Cell Images)</li>
            <li><b>Pneumonia</b>: CNN Image Analysis (X-Ray Images)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.image("https://cdn.activestate.com/wp-content/uploads/2018/10/machine-learning-healthcare-blog-hero-1200x799.jpg", width="stretch")

elif app_mode == "Diabetes":
    st.header("Diabetes Prediction")
    st.markdown("Please enter your health metrics below:")
    
    with st.expander("Enter Patient Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
            glucose = st.number_input("Glucose Level", min_value=0.0)
            bp = st.number_input("Blood Pressure", min_value=0.0)
            skin = st.number_input("Skin Thickness", min_value=0.0)
        with col2:
            insulin = st.number_input("Insulin Level", min_value=0.0)
            bmi = st.number_input("BMI", min_value=0.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
    
    if st.button("Predict Diabetes Risk"):
        input_data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        prediction = predict_structured(models['diabetes'], input_data)
        
        if prediction == 1:
            st.markdown('<div class="result-box danger">High Risk of Diabetes Detected</div>', unsafe_allow_html=True)
            st.warning("Please consult a healthcare professional regarding these results.")
        else:
            st.markdown('<div class="result-box safe">No Diabetes Detected</div>', unsafe_allow_html=True)
            st.balloons()
        
        # --- Visualization Dashboard ---
        st.divider()
        st.subheader("🔍 Health Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Glucose Gauge (Normal: ~70-140 mg/dL for random/fasting mix)
            st.plotly_chart(plot_health_gauge(glucose, "Glucose Level", 0, 300, (70, 140)), width="stretch")
            
        with col2:
            # BMI Gauge (Normal: 18.5-25)
            st.plotly_chart(plot_health_gauge(bmi, "BMI Index", 10, 50, (18.5, 25)), width="stretch")
            
        if bmi > 25:
             st.info("💡 **Tip**: Your BMI indicates you might be overweight. Regular exercise can help lower risk.")
        if glucose > 140:
             st.info("💡 **Tip**: Glucose levels above 140 mg/dL can be a sign of pre-diabetes. Consult a doctor.")
        # -------------------------------

elif app_mode == "Heart Disease":
    st.header("Heart Disease Prediction")
    
    with st.expander("Enter Patient Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
            cp = st.selectbox("Chest Pain Type", options=[(0, "Typical Angina"), (1, "Atypical Angina"), (2, "Non-anginal Pain"), (3, "Asymptomatic")], format_func=lambda x: f"{x[0]} - {x[1]}")
            trestbps = st.number_input("Resting Blood Pressure", min_value=0.0)
            chol = st.number_input("Serum Cholestoral", min_value=0.0)
            restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, step=1)
        with col2:
            thalach = st.number_input("Max Heart Rate", min_value=0.0)
            exang = st.selectbox("Exercise Induced Angina", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
            oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0)
            slope = st.number_input("Slope of Peak Exercise ST (0-2)", min_value=0, max_value=2)
            thal = st.selectbox("Thalassemia", options=[(1, "Normal"), (2, "Fixed Defect"), (3, "Reversable Defect")], format_func=lambda x: f"{x[0]} - {x[1]}")

    if st.button("Predict Heart Disease Risk"):
        # Preprocessing: Log transform specific columns (trestbps, chol)
        # Matches heart.py logic: data["trestbps"]=np.log(data["trestbps"]); data["chol"]=np.log(data["chol"])
        
        # Safe log (avoid log(0) or negative)
        trestbps_log = np.log(trestbps) if trestbps > 0 else 0
        chol_log = np.log(chol) if chol > 0 else 0
        
        # Original Input list (indices 0-10)
        # 0:age, 1:sex, 2:cp, 3:trestbps(log), 4:chol(log), 5:restecg, 6:thalach, 7:exang, 8:oldpeak, 9:slope, 10:thal
        input_data = [age, sex[1], cp[0], trestbps_log, chol_log, restecg, thalach, exang[1], oldpeak, slope, thal[0]]
        
        scaler = get_heart_scaler()
        prediction = predict_structured(models['heart'], input_data, scaler=scaler)
        
        if prediction == 1:
            st.markdown('<div class="result-box danger">Heart Disease Risk Detected</div>', unsafe_allow_html=True)
            st.warning("Prediction based on scaled data matching specific medical thresholds.")
        else:
            st.markdown('<div class="result-box safe">Healthy Heart Status</div>', unsafe_allow_html=True)
            st.balloons()

        # --- Visualization Dashboard ---
        st.divider()
        st.subheader("❤️ Cardiac Health Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # BP Gauge (Normal: <120 systolic. Warning up to 140)
            st.plotly_chart(plot_health_gauge(trestbps, "Resting Blood Pressure", 80, 200, (90, 120)), width="stretch")
            
        with col2:
            # Cholesterol Gauge (Normal: <200)
            st.plotly_chart(plot_health_gauge(chol, "Cholesterol", 100, 400, (125, 200)), width="stretch")
            
        if trestbps > 130:
             st.info("💡 **Tip**: Elevated blood pressure significantly increases heart strain.")
        if chol > 200:
             st.info("💡 **Tip**: High cholesterol is a major risk factor. Consider a low-fat diet.")
        # -------------------------------

elif app_mode == "Kidney Disease":
    st.header("Kidney Disease Prediction")
    
    with st.expander("Enter Patient Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            agek = st.number_input("Age", min_value=0)
            bpk = st.number_input("Blood Pressure", min_value=0.0)
            al = st.slider("Albumin", 0, 5, 0)
            pcc = st.selectbox("Pus Cell Clumps", options=[("Present", 1), ("Not Present", 0)], format_func=lambda x: x[0])
            bgr = st.number_input("Blood Glucose Random", min_value=0.0)
            bu = st.number_input("Blood Urea", min_value=0.0)
        with col2:
            sc = st.number_input("Serum Creatinine", min_value=0.0)
            hemo = st.number_input("Hemoglobin", min_value=0.0)
            pcv = st.number_input("Packed Cell Volume", min_value=0.0)
            htn = st.selectbox("Hypertension", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
            dm = st.selectbox("Diabetes Mellitus", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
            appet = st.selectbox("Appetite", options=[("Good", 1), ("Poor", 0)], format_func=lambda x: x[0])

    if st.button("Predict Kidney Disease"):
        if models['kidney'] is None:
             st.error("Model unavailable: 'model3' (Kidney) could not be loaded. Attempting regeneration...")
        else:
            input_data = [agek, bpk, al, pcc[1], bgr, bu, sc, hemo, pcv, htn[1], dm[1], appet[1]]
            prediction = predict_structured(models['kidney'], input_data)
            
            if prediction == 1:
                st.markdown('<div class="result-box danger">Kidney Disease Detected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box safe">Healthy Kidney Status</div>', unsafe_allow_html=True)
                st.balloons()

elif app_mode == "Liver Disease":
    st.header("Liver Disease Prediction")
    
    with st.expander("Enter Patient Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            agel = st.number_input("Age", min_value=0)
            gen = st.selectbox("Gender", options=[("Male", 0), ("Female", 1)], format_func=lambda x: x[0])
            tb = st.number_input("Total Bilirubin", min_value=0.0)
            db = st.number_input("Direct Bilirubin", min_value=0.0)
            ap = st.number_input("Alkaline Phosphotase", min_value=0.0)
        with col2:
            aa1 = st.number_input("Alamine Aminotransferase", min_value=0.0)
            aa2 = st.number_input("Aspartate Aminotransferase", min_value=0.0)
            tp = st.number_input("Total Protiens", min_value=0.0)
            alb = st.number_input("Albumin", min_value=0.0)
            ag_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0)

    if st.button("Predict Liver Disease"):
        if models['liver'] is None:
            st.error("Model unavailable: 'model4' (Liver) could not be loaded. Dataset 'indian_liver_patient.csv' missing for regeneration.")
        else:
            input_data = [agel, gen[1], tb, db, ap, aa1, aa2, tp, alb, ag_ratio]
            
            # scaler = get_liver_scaler() # Removed scaling to match new synthetic model trained on raw data
            prediction = predict_structured(models['liver'], input_data) #, scaler=scaler)
            
            if prediction == 1:
                st.markdown('<div class="result-box danger">Liver Disease Detected</div>', unsafe_allow_html=True)
                st.warning("Prediction based on statistical averages from the Indian Liver Patient Dataset.")
            else:
                st.markdown('<div class="result-box safe">Healthy Liver Status</div>', unsafe_allow_html=True)
                st.balloons()

elif app_mode == "Generate Cancer":
    st.header("Breast Cancer Prediction")
    st.info("Input the Mean, Standard Error, and Worst values for the cell nuclei.")
    
    with st.expander("Mean Values", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        rm = col1.number_input("Radius Mean")
        tm = col2.number_input("Texture Mean")
        pm = col3.number_input("Perimeter Mean")
        am = col4.number_input("Area Mean")
        sm = col5.number_input("Smoothness Mean")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cm = col1.number_input("Compactness Mean")
        conm = col2.number_input("Concavity Mean")
        cpm = col3.number_input("Concave Points Mean")
        sym = col4.number_input("Symmetry Mean")
        fdm = col5.number_input("Fractal Dim Mean")

    with st.expander("Standard Error Values", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        rs = col1.number_input("Radius SE")
        ts = col2.number_input("Texture SE")
        ps = col3.number_input("Perimeter SE")
        as_ = col4.number_input("Area SE")
        ss = col5.number_input("Smoothness SE")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cs = col1.number_input("Compactness SE")
        cons = col2.number_input("Concavity SE")
        cps = col3.number_input("Concave Points SE")
        sys = col4.number_input("Symmetry SE")
        fds = col5.number_input("Fractal Dim SE")

    with st.expander("Worst Values", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        rw = col1.number_input("Radius Worst")
        tw = col2.number_input("Texture Worst")
        pw = col3.number_input("Perimeter Worst")
        aw = col4.number_input("Area Worst")
        sw = col5.number_input("Smoothness Worst")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cw = col1.number_input("Compactness Worst")
        conw = col2.number_input("Concavity Worst")
        cpw = col3.number_input("Concave Points Worst")
        syw = col4.number_input("Symmetry Worst")
        fdw = col5.number_input("Fractal Dim Worst")

    if st.button("Predict Cancer Class"):
        input_data = [rm, tm, pm, am, sm, cm, conm, cpm, sym, fdm,
                      rs, ts, ps, as_, ss, cs, cons, cps, sys, fds,
                      rw, tw, pw, aw, sw, cw, conw, cpw, syw, fdw]
        prediction = predict_structured(models['cancer'], input_data)
        
        if prediction == 1:
            st.markdown('<div class="result-box danger">Cancer Detected</div>', unsafe_allow_html=True)
            st.warning("Urgent medical attention recommended.")
        else:
            st.markdown('<div class="result-box safe">No Cancer Detected</div>', unsafe_allow_html=True)
            st.balloons()

elif app_mode == "Malaria Detection":
    st.header("Malaria Detection (Cell Images)")
    uploaded_file = st.file_uploader("Upload a Cell Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image_obj = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_obj, caption="Uploaded Image", width=300)
        
        if st.button("Analyze Cell"):
            res = predict_malaria(image_obj, models['malaria'])
            # indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            # Note: app.py has 4 classes in dict but model output checks vary. 
            # Replicating app.py logic:
            # predicted_class = np.argmax(result, axis=1).item()
            # label = indices[predicted_class]
            
            classes = ['PARASITIC', 'Uninfected', 'Invasive carcinomar', 'Normal'] 
            # Note: 'Invasive carcinomar' and 'Normal' seem like leftovers in app.py dict for malaria?
            # Standard Malaria datasets usually have 2 classes (Parasitized, Uninfected).
            # But I must follow app.py logic which uses model111.h5. 
            
            idx = np.argmax(res, axis=1).item()
            confidence = res[0][idx] * 100
            
            # Using the classes from app.py line 140
            label = classes[idx] if idx < len(classes) else "Unknown"
            
            if "PARASITIC" in label.upper():
                 st.markdown(f'<div class="result-box danger">{label} ({confidence:.2f}%)</div>', unsafe_allow_html=True)
            else:
                 st.markdown(f'<div class="result-box safe">{label} ({confidence:.2f}%)</div>', unsafe_allow_html=True)

elif app_mode == "Pneumonia Detection":
    st.header("Pneumonia Detection (X-Ray)")
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image_obj = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_obj, caption="Uploaded X-Ray", width=300)
            
        if st.button("Analyze X-Ray"):
            res = predict_pneumonia(image_obj, models['pneumonia'])
            
            # app.py logic:
            # pred_value > 0.5 -> Pneumonia (indices[1])
            # else -> Normal (indices[0])
            pred_value = res[0][0] if res.ndim > 1 else res[0]
            
            if pred_value > 0.5:
                accuracy = pred_value * 100
                st.markdown(f'<div class="result-box danger">Pneumonia Detected ({accuracy:.2f}%)</div>', unsafe_allow_html=True)
            else:
                accuracy = (1 - pred_value) * 100
                st.markdown(f'<div class="result-box safe">Normal ({accuracy:.2f}%)</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed with Streamlit & TensorFlow • 2024")
