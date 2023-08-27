#Importing setup
import cv2
import os
import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

#Data organization setup
image_dir = 'datasets/'

no_tumours = os.listdir(image_dir + 'no')
yes_tumours = os.listdir(image_dir + 'yes')
dataset = []
label = []
input_size = 64

for i, image_name in enumerate(no_tumours):
    if image_name.split('.')[1]=='jpg':
        image = cv2.imread(image_dir + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size, input_size))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumours):
    if image_name.split('.')[1]=='jpg':
        image = cv2.imread(image_dir + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size, input_size))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state = 0)
#note that Reshape (e.g print(x_train.shape))= (n, image_width, image_height, n_channel)

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes = 2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 2)

#Model Building
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape = (input_size, input_size, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

    tf.keras.layers.Conv2D(32, (3,3), kernel_initializer = 'he_uniform'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

    tf.keras.layers.Conv2D(32, (3,3), kernel_initializer = 'he_uniform'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation('softmax')
])

# Binary CrossEntropy = 1, sigmoid
# Categorical Cross Entropy = 2, softmax

#Model Compilation and Training
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 16, verbose = True, epochs = 10, 
validation_data = (x_test, y_test), shuffle = False)

model.save('BrainTumour10EpochsCategorical.h5')