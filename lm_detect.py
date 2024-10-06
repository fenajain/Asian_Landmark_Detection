import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import os
import base64

# --- GENERAL SETTINGS ---
PAGE_TITLE = "Asian Landmark Detection"
PAGE_ICON = "https://www.gstatic.com/webp/gallery/2.jpg"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

def add_bg_from_local(image_files):
    with open(image_files[0], "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(image_files[1], "rb") as image_file:
        encoded_string1 = base64.b64encode(image_file.read())
    st.markdown(
    """
    <style>
      .stApp {
          background-image: url(data:image/png;base64,"""+encoded_string.decode()+""");
          background-size: cover;
      }
      .css-1avcm0n.e8zbici2 {
        background-image: url(data:image/png;base64,"""+encoded_string1.decode()+""");
        background-size: cover;
        background-repeat: no-repeat;
      }
    </style>"""
    ,
    unsafe_allow_html=True
    )
add_bg_from_local([r'dark.png',r'dark.png'])

model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
# model_url = 'on_device_vision_classifier_landmarks_classifier_asia_V1_1'

# label_url = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
labels = 'landmark_detection_labels.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))

def image_processing(image):
    img_shape = (321, 321)

    # Load the TensorFlow Hub model with the correct input shape and specify the output_key
    hub_layer = hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")

    # Open and preprocess the image
    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img  # Save the original image for displaying later
    img = np.array(img) / 255.0  # Normalize the image
    img = img[np.newaxis]  # Add batch dimension

    # Perform prediction using the model
    logits = hub_layer(img)

    # Get the predicted label using the logits
    predicted_label = labels[np.argmax(logits)]
    
    return predicted_label, img1


def get_map(loc):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(loc)
    return location.address,location.latitude, location.longitude

# Create a directory to save uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def run():
    st.title("Asian Landmark Recognition")
    img = PIL.Image.open('logo.png')
    img = img.resize((256,256))
    st.image(img)
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
    if img_file is not None:
        save_image_path = os.path.join(UPLOAD_DIR, img_file.name)
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        result_placeholder = st.empty()
        with st.spinner('Running the analysis...'):
            try:
                prediction, image = image_processing(save_image_path)
                st.image(image)
                st.write('') # Top provide a gap
                st.header("📍 **Predicted Landmark is: " + prediction + '**')
            except Exception as e:
                st.warning(e)
        result_placeholder.write()
        try:
            address, latitude, longitude = get_map(prediction)
            st.success('Address: '+address )
            loc_dict = {'Latitude':latitude,'Longitude':longitude}
            st.write('') # Top provide a gap
            st.subheader('✅ **Latitude & Longitude of '+prediction+'**')
            st.json(loc_dict)
            data = [[latitude,longitude]]
            df = pd.DataFrame(data, columns=['lat', 'lon'])
            st.subheader('✅ **'+prediction +' on the Map**'+'🗺️')
            st.map(df)
        except Exception as e:
            st.warning("No address found!!")
run()
