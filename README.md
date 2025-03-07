![dog project (1) (1) (1) (1)](https://github.com/user-attachments/assets/7237939d-8e43-4b4a-9943-81986a30110e)

# Pawdentify 🐾

Project Demo: https://drive.google.com/file/d/1coR61nP7s_TRYDyEHFBukn4y6wYZWGdB/view?usp=sharing

## Dog Breed Classification using Deep Learning

Pawdentify is a deep learning application that can identify dog breeds from images with high accuracy. Using convolutional neural networks (CNN) and transfer learning, this project can classify dogs into one of 120 different breeds.

## 📊 Project Overview

- **Accuracy Rate**: 78% across 120 dog breed classes
- **Dataset Size**: 10,222 images
- **Model Architecture**: Leverages MobileNetV2 through transfer learning
- **Technologies**: TensorFlow, NumPy, Pandas, MatPlotLib

## 🔧 Features

- Accurate classification of 120 dog breeds from images
- Pre-processed and optimized dataset for training
- Transfer learning implementation using MobileNetV2 architecture
- Comprehensive data visualization for analysis

## 📋 Requirements

```
tensorflow>=2.4.0
numpy>=1.19.5
pandas>=1.2.0
matplotlib>=3.3.0
pillow>=8.0.0
scikit-learn>=0.24.0
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pawdentify.git
cd pawdentify

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

```python
python train.py --data_dir /path/to/dog/images --epochs 50 --batch_size 32
```

### Making Predictions

```python
python predict.py --image /path/to/dog/image.jpg
```

### Using the Pre-trained Model

```python
from pawdentify import DogClassifier

# Load the model
classifier = DogClassifier(model_path='models/pawdentify_model.h5')

# Predict breed from image
breed, confidence = classifier.predict('path/to/dog/image.jpg')
print(f"Predicted breed: {breed} with {confidence:.2f}% confidence")
```

## 🧠 Model Architecture

Pawdentify utilizes transfer learning with MobileNetV2 as the base model. This architecture was chosen for its balance of accuracy and efficiency. The pre-trained MobileNetV2 layers extract powerful features from dog images, which are then fed into our custom classification layers trained specifically for dog breed identification.

## 📈 Performance

The model achieves 78% accuracy across 120 dog breeds, which significantly outperforms random chance (0.83%) and is comparable to specialized dog breed identification systems.

Tanishq Jain
