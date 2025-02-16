import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Dataset
train_dir = "dataset/train"  # Change to your dataset path , you can download data from Kaggle , link in ReadME
test_dir = "dataset/test"

img_size = 128 
batch_size = 32

# Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Train Model
epochs = 5  # Change for more training
model.fit(train_generator, epochs=epochs, validation_data=test_generator)

# Save Model
model.save("crop_disease_model.h5")
print("✅ Model trained and saved as 'crop_disease_model.h5'")

# Load & Predict on New Image with File Import
def predict_disease():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        print("No file selected!")
        return
    model = load_model("crop_disease_model.h5")

    img = cv2.imread(file_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_labels = list(train_generator.class_indices.keys())
    
    result_text = f"🟢 Predicted Disease: {class_labels[class_idx]} (Confidence: {np.max(prediction)*100:.2f}%)"
    print(result_text)

    # Display Image and Result
    img_display = cv2.imread(file_path)
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title(result_text)
    plt.axis("off")
    plt.show()

# Tkinter to Select Image and Predict
root = tk.Tk()
root.withdraw()  # Hide main window
predict_disease()
