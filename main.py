import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import tensorflow_hub as hub

model_url = "https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1"
labels = "landmark_detection_labels.csv"

img_shape = (321,321)
classifier = tf.keras.Sequential([hub.KerasLayer(model_url, input_shape=img_shape+(3,), output_key = "predictions:logits")])

df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))

# Testing on Image
img = PIL.Image.open("s1.jpg")
img = img.resize(img_shape)
img = np.array(img)/255.0
img = img[np.newaxis]
result = classifier.predict(img)
print(labels[np.argmax(result)])
