from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import io
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

labels_csv = pd.read_csv("labels.csv")
labels = labels_csv["breed"].to_numpy()
unique_breeds = np.unique(labels)


# Load the model with custom objects
try:
    model = keras.models.load_model("20250215-19401739648421-full_model.keras")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevents crashing if the model fails to load

# Replace with actual dog breed class names
class_names = unique_breeds

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Ensure size matches model input
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
    img = preprocess_image(img)

    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    return jsonify({"breed": predicted_class})

# Add CORS headers manually
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
