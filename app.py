import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
import efficientnet.tfkeras as efficientnet
import matplotlib.pyplot as plt

class CarClassificationModel:
    def __init__(self):
        self.img_width = 224
        self.img_height = 224
        self.batch_size = 32
        self.models = {}

    def setup_paths(self, base_path):
        self.paths = {
            'base': base_path,
            'train': os.path.join(base_path, 'train'),
            'test': os.path.join(base_path, 'test'),
            'models': os.path.join(base_path, 'saved_models')
        }
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def setup_data_generators(self):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            zoom_range=0.2,
            rotation_range=5,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.paths['train'],
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        self.validation_generator = test_datagen.flow_from_directory(
            self.paths['test'],
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        self.class_indices = self.train_generator.class_indices
        self.classes = list(self.class_indices.keys())

    def create_efficientnet_model(self):
        base_model = efficientnet.EfficientNetB1(weights=None, include_top=False)
        x = GlobalAveragePooling2D()(base_model.output)
        predictions = Dense(len(self.class_indices), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

        # Load pre-trained weights if available
        weights_path = os.path.join(self.paths['models'], 'efficientnet_weights.h5')
        if os.path.exists(weights_path):
            model.load_weights(weights_path)

        self.models['efficientnet'] = model
        return model

    def predict_image(self, image_path, model_name='efficientnet'):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Please create it first.")

        img = Image.open(image_path)
        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.models[model_name].predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence_score = predictions[0][predicted_class_idx]

        predicted_class = list(self.class_indices.keys())[list(self.class_indices.values()).index(predicted_class_idx)]

        return predicted_class, confidence_score

    def predict_class(self, model_name='efficientnet'):
        image_batch, classes_batch = next(self.validation_generator)

        predicted_batch = self.models[model_name].predict(image_batch)

        for i in range(3):
            image = image_batch[i]
            pred = predicted_batch[i]

            the_pred = np.argmax(pred)
            predicted_class_name = list(self.class_indices.keys())[the_pred]

            val_pred = max(pred)

            the_class_idx = np.argmax(classes_batch[i])
            true_class_name = list(self.class_indices.keys())[the_class_idx]

            isTrue = (the_pred == the_class_idx)

            plt.figure(i)
            plt.title(f"{isTrue} - True: {true_class_name} - Predicted: {predicted_class_name} - Probability: {val_pred:.4f}")
            plt.imshow(image)

# Streamlit interface
st.title('Car Classification App')

# Initialize model
classifier = CarClassificationModel()

# Setup paths - replace with your actual path
base_path = "C:/Users/shukl/Documents/College/!ML/car_data-20241113T061837Z-001/car_data"
classifier.setup_paths(base_path)

# Setup data generators
classifier.setup_data_generators()

# Create models
classifier.create_efficientnet_model()

# Streamlit interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox('Select Model', ['EfficientNet'])

if uploaded_file is not None:
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    predicted_class, confidence = classifier.predict_image(temp_path, model_name='efficientnet')

    st.image(uploaded_file, caption='Uploaded Image')
    st.write(f'Predicted class: {predicted_class}')
    st.write(f'Confidence: {confidence:.2%}')

    # Optionally, show the prediction class visualization
    classifier.predict_class(model_name='efficientnet')
    st.pyplot(plt)

    # Clean up
    os.remove(temp_path)