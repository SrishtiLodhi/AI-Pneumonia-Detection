# AI Pneumonia Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange) ![Keras](https://img.shields.io/badge/Keras-2.14-red)

## 🚀 Project Overview
AI Pneumonia Detection is a **deep learning-based system** to classify chest X-ray images as either **Normal** or **Pneumonia** (Bacterial/Viral). This project leverages **Convolutional Neural Networks (CNNs)** for accurate detection to aid early diagnosis.

## 🗂 Dataset
- **Source:** Chest X-ray images organized into `train`, `val`, and `test` directories.
- **Classes:** `NORMAL`, `PNEUMONIA` (Bacterial & Viral)
  
## 🧠 Model Architecture
- Built using **TensorFlow/Keras**
- CNN with multiple convolutional and pooling layers
- **Activation:** ReLU, output layer with Softmax for multi-class classification
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam

## ⚡ Features
- Train and validate CNN on chest X-ray dataset
- Evaluate model performance using **accuracy** and **loss**
- Save and load trained models (`pneumonia_model.keras`)
- Scripts for **training** and **evaluation**:
  - `train.py` – trains the model  
  - `evaluate.py` – evaluates accuracy on test set  
  - `model.py` – defines the CNN architecture  

## 🏁 Getting Started

### Download Instructions
The dataset is available on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia):

1. **Manually:**
   - Download the `.zip` file from Kaggle.
   - Extract it and place the contents into the `data/` folder:

2. **Using Kaggle CLI (optional, fully reproducible):**
```bash
pip install kaggle
mkdir data && cd data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
cd ..

### Clone Repository
```bash
git clone https://github.com/SrishtiLodhi/AI-Pneumonia-Detection.git
cd AI-Pneumonia-Detection
