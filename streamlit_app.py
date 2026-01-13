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
@st.cache_resource
def load_models():
    models = {}
    try:
        models['malaria'] = load_model('model111.h5')
        models['pneumonia'] = load_model('my_model.h5')
        models['diabetes'] = joblib.load("model1")
        models['cancer'] = joblib.load("model")
        models['kidney'] = joblib.load("model3")
        models['liver'] = joblib.load("model4")
        models['heart'] = joblib.load("model2")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models = load_models()

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

def predict_structured(model, input_list):
    to_predict = np.array(input_list).reshape(1, len(input_list))
    result = model.predict(to_predict)
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
    st.image("https://cdn.activestate.com/wp-content/uploads/2018/10/machine-learning-healthcare-blog-hero-1200x799.jpg", use_column_width=True)

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
        input_data = [age, sex[1], cp[0], trestbps, chol, restecg, thalach, exang[1], oldpeak, slope, thal[0]]
        prediction = predict_structured(models['heart'], input_data)
        
        if prediction == 1:
            st.markdown('<div class="result-box danger">Heart Disease Risk Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box safe">Healthy Heart Status</div>', unsafe_allow_html=True)
            st.balloons()

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
        input_data = [agel, gen[1], tb, db, ap, aa1, aa2, tp, alb, ag_ratio]
        prediction = predict_structured(models['liver'], input_data)
        
        if prediction == 1:
            st.markdown('<div class="result-box danger">Liver Disease Detected</div>', unsafe_allow_html=True)
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
