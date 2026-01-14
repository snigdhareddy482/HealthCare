# Important Modules
from flask import Flask, render_template, url_for, flash, redirect, request, send_from_directory
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Models
# Using global variables for models. Note: In production, consider lazy loading or a serving infrastructure.
try:
    print("Loading models...")
    model_malaria = load_model('model111.h5', compile=False)
    model_pneumonia = load_model('my_model.h5', compile=False)
    
    # Models for structured data (Heart, Diabetes, etc.)
    # These rely on joblib for sklearn models
    model_diabetes = joblib.load("model1")
    model_cancer = joblib.load("model")
    model_kidney = joblib.load("model3")
    model_liver = joblib.load("model4")
    model_heart = joblib.load("model2")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    # Continue implementation to allow debugging, but predictions will fail


# Malaria Prediction
def predict_malaria(full_path):
    img = image.load_img(full_path, target_size=(50, 50, 3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize

    predicted = model_malaria.predict(img_array)
    return predicted

# Pneumonia Prediction
def predict_pneumonia(full_path):
    img = image.load_img(full_path, target_size=(64, 64, 3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predicted = model_pneumonia.predict(img_array)
    return predicted


# Structured Data Prediction
def predict_structured(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    
    if size == 8: # Diabetes
        result = model_diabetes.predict(to_predict)
    elif size == 30: # Cancer
        result = model_cancer.predict(to_predict)
    elif size == 12: # Kidney
        result = model_kidney.predict(to_predict)
    elif size == 10: # Liver
        result = model_liver.predict(to_predict)
    elif size == 11: # Heart
        result = model_heart.predict(to_predict)
    else:
        return None

    return result[0]

# --- Routes ---

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

# Disease Page Routes
@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

@app.route("/malaria")
def malaria():
    return render_template("index.html")

@app.route("/pneumonia")
def pneumonia():
    return render_template("index2.html")


# Malaria Prediction Endpoint
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            if 'image' not in request.files:
                flash('No file part', 'danger')
                return redirect(request.url)
            
            file = request.files['image']
            if file.filename == '':
                flash('No selected file', 'danger')
                return redirect(request.url)

            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = predict_malaria(full_name)
            
            # Modernize numpy: use .item() instead of asscalar if needed, though argmax returns int usually.
            predicted_class = np.argmax(result, axis=1).item()
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            
            return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for("malaria"))

# Pneumonia Prediction Endpoint
@app.route('/upload11', methods=['POST', 'GET'])
def upload11_file():
    if request.method == 'GET':
        return render_template('index2.html')
    else:
        try:
             if 'image' not in request.files:
                flash('No file part', 'danger')
                return redirect(request.url)
            
             file = request.files['image']
             if file.filename == '':
                flash('No selected file', 'danger')
                return redirect(request.url)

             full_name = os.path.join(UPLOAD_FOLDER, file.filename)
             file.save(full_name)
             
             indices = {0: 'Normal', 1: 'Pneumonia'}
             result = predict_pneumonia(full_name)
             
             # Assuming binary classification output structure from original code logic
             # The original code logic was a bit ambiguous: if(result>50)
             # We'll assume result is a probability of class 1.
             
             # Safeguard: check shape.
             pred_value = result[0][0] if result.ndim > 1 else result[0]
             
             if pred_value > 0.5: # Thresholding at 0.5 usually standard for sigmoid
                 label = indices[1]
                 accuracy = round(pred_value * 100, 2)
             else:
                 label = indices[0]
                 accuracy = round((1 - pred_value) * 100, 2)
                 
             return render_template('predict1.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for("pneumonia"))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# Structured Data Prediction Endpoint
@app.route('/result', methods=["POST"])
def result():
    if request.method == 'POST':
        try:
            to_predict_list = request.form.to_dict()
            to_predict_list = list(to_predict_list.values())
            to_predict_list = list(map(float, to_predict_list))
            
            result_val = predict_structured(to_predict_list, len(to_predict_list))
            
            if result_val is not None:
                 if int(result_val) == 1:
                     prediction = 'Sorry ! Suffering'
                 else:
                     prediction = 'Congrats ! you are Healthy'
            else:
                 prediction = "Error: Invalid input size"

        except Exception as e:
            prediction = f"Error during prediction: {e}"

        return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
