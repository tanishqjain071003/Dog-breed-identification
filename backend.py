import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable for the model
model = None

# Initialize labels
try:
    labels_csv = pd.read_csv("labels.csv")
    labels = labels_csv["breed"].to_numpy()
    unique_breeds = np.unique(labels)
    class_names = unique_breeds
    print("✅ Labels loaded successfully!")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    # Fallback to some default breeds if needed
    class_names = ["Labrador", "Golden Retriever", "German Shepherd"]  # Example fallback

# Load TensorFlow and model only when needed
def load_model():
    global model
    if model is None:
        try:
            import tensorflow as tf
            from tensorflow import keras
            model = keras.models.load_model("20250215-19401739648421-full_model.keras")
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    return True

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Load model on demand
    if not load_model():
        return jsonify({"error": "Model failed to load"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        from tensorflow.keras.preprocessing import image
        file = request.files["file"]
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img = preprocess_image(img)

        predictions = model.predict(img)
        predicted_class = class_names[np.argmax(predictions)]

        return jsonify({"breed": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Add CORS headers manually
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS, GET"
    return response

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
