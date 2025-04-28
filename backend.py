import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
model = None
class_names = None

# Initialize labels
def load_labels():
    global class_names
    try:
        labels_csv = pd.read_csv("labels.csv")
        labels = labels_csv["breed"].to_numpy()
        class_names = np.unique(labels)
        print("✅ Labels loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        # Fallback to some default breeds if needed
        class_names = ["Labrador", "Golden Retriever", "German Shepherd"]  # Example fallback
        return False

# Load TensorFlow and model only when needed
def load_model():
    global model
    if model is None:
        try:
            # Import TensorFlow here to avoid immediate load issues
            import tensorflow as tf
            from tensorflow import keras

            # Load the model directly from the file in the repository
            model = keras.models.load_model("20250215-19401739648421-full_model.keras")
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    return True

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Ensure size matches model input
    img = np.array(img) / 255.0    # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Root endpoint for health check
@app.route("/", methods=["GET"])
def root():
    return "Dog Breed Classifier API is running!"

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "labels_loaded": class_names is not None,
        "model_loaded": model is not None
    })

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Load labels if not loaded
    if class_names is None:
        if not load_labels():
            return jsonify({"error": "Failed to load breed labels"}), 500

    # Check for file
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Load model on demand
        if not load_model():
            return jsonify({"error": "Model failed to load"}), 500

        # Import here to avoid immediate loading issues
        from tensorflow.keras.preprocessing import image
        
        # Process the image
        file = request.files["file"]
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img_array = preprocess_image(img)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        return jsonify({"breed": predicted_class})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Add CORS headers manually
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS, GET"
    return response

# Run the Flask app
if __name__ == "__main__":
    # Load labels at startup
    load_labels()
    
    # Get port from environment variable (for Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
