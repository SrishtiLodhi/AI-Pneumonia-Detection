import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Dataset path
test_dir = "data/chest_xray/test"

# Image preprocessing
test_gen = ImageDataGenerator(rescale=1./255)

test_batches = test_gen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Load trained model
model = load_model("pneumonia_model.keras")

# Predictions
predictions = model.predict(test_batches)
predicted_classes = (predictions > 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(test_batches.classes, predicted_classes)

# Seaborn visualization
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()