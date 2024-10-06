import streamlit as st
import os
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
# model_url = 'on_device_vision_classifier_landmarks_classifier_asia_V1_1'

# label_url = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
labels = 'landmark_detection_labels.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))

def image_processing(image):
    img_shape = (321, 321)
    
    # Load the classifier model from TensorFlow Hub
    classifier = tf.keras.Sequential([
        hub.KerasLayer(model_url, input_shape=img_shape + (3,))
    ])
    
    # Process the image
    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img
    img = np.array(img) / 255.0
    img = img[np.newaxis]
    
    # Perform prediction
    result = classifier.predict(img)
    
    # Ensure the output matches the expected format
    return labels[np.argmax(result)], img1

def get_map(loc):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(loc)
    return location.address,location.latitude, location.longitude

import os  # Add this import for directory checks

def run():
    st.title("Landmark Recognition")
    img = PIL.Image.open('logo.png')
    img = img.resize((256, 256))
    st.image(img)
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
    
    if img_file is not None:
        # Define the directory to save uploaded images
        save_dir = './Uploaded_Images/'
        
        # Check if the directory exists; if not, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Full path where the uploaded image will be saved
        save_image_path = os.path.join(save_dir, img_file.name)
        
        # Save the uploaded image to the directory
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        # Process and display the image
        prediction, image = image_processing(save_image_path)
        st.image(image)
        st.header("📍 **Predicted Landmark is: " + prediction + '**')
        
        try:
            address, latitude, longitude = get_map(prediction)
            st.success('Address: ' + address)
            loc_dict = {'Latitude': latitude, 'Longitude': longitude}
            st.subheader('✅ **Latitude & Longitude of ' + prediction + '**')
            st.json(loc_dict)

            # Display location on map
            data = [[latitude, longitude]]
            df = pd.DataFrame(data, columns=['lat', 'lon'])
            st.subheader('✅ **' + prediction + ' on the Map**' + '🗺️')
            st.map(df)
        
        except Exception as e:
            st.warning("No address found!!")

run()
