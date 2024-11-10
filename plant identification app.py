!pip install streamlit
!pip install tensorflow
!pip install pillow
!pip install numpy

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load and preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to fit the model input size (224x224 for MobileNetV2)
    img_array = np.array(img) / 255.0  # Normalize the image (if required by the model)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to load the pre-trained MobileNetV2 model
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")  # Using MobileNetV2 pre-trained model
    return model

# Function to identify the plant using the model
def identify_plant(image):
    model = load_model()
    img_array = preprocess_image(image)
    
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    plant_name = decoded_predictions[0][1]  # Top prediction name (e.g., "rose")
    confidence = decoded_predictions[0][2]  # Confidence level of the top prediction
    return plant_name, confidence

# Function to get care tips for the identified plant
def get_care_tips(plant_name):
    care_tips = {
        "rose": "Water regularly, keep in full sun, and prune dead flowers.",
        "sunflower": "Needs full sun and regular watering. Grows best in well-drained soil.",
        "cactus": "Water infrequently, prefers dry conditions, and indirect sunlight.",
        "tulip": "Plant in well-drained soil, needs moderate water and sunlight.",
        "orchid": "Prefers indirect light and needs to be watered once a week."
    }
    return care_tips.get(plant_name.lower(), "Care tips not available.")

# Function to suggest compatible plants
def suggest_compatible_plants(plant_name):
    compatible_plants = {
        "rose": ["lavender", "thyme", "geranium"],
        "sunflower": ["zinnia", "cosmos", "marigold"],
        "cactus": ["succulent", "agave", "aloe vera"],
        "tulip": ["daffodil", "hyacinth", "iris"],
        "orchid": ["ferns", "air plants", "bamboo"]
    }
    return compatible_plants.get(plant_name.lower(), ["No compatible plants found."])

# Streamlit UI
st.title("Plant Identification App")

# Upload an image of a plant
uploaded_image = st.file_uploader("Upload a Plant Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Plant Image", use_column_width=True)

    # Identify the plant
    plant_name, confidence = identify_plant(image)
    
    # Display results
    st.subheader(f"Identified Plant: {plant_name}")
    st.write(f"Confidence Level: {confidence * 100:.2f}%")

    # Get and display care tips
    care_tips = get_care_tips(plant_name)
    st.subheader(f"Care Tips for {plant_name}:")
    st.write(care_tips)

    # Suggest compatible plants
    compatible_plants = suggest_compatible_plants(plant_name)
    st.subheader(f"Compatible Plants with {plant_name}:")
    st.write(", ".join(compatible_plants))
