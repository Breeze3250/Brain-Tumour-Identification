#Importing Setup
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np

#Model and Imaging Setup
model = tf.keras.models.load_model('BrainTumour10EpochsCategorical.h5')
image = cv2.imread('C:\\Users\\brend\Downloads\\archive\\pred\\pred5.jpg')

image = Image.fromarray(image)
image = image.resize((64, 64))
image = np.array(image)
input_image = np.expand_dims(image, axis = 0)

result =  model.predict(input_image)
result_final = np.argmax(result, axis = 1)
print(result_final)