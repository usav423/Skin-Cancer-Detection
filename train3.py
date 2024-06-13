import numpy as np
from tensorflow import keras
from keras.applications import ResNet50
import matplotlib.pyplot as plt
import os
import cv2
import random
import sklearn.model_selection as model_selection
import datetime
from keras import layers
categories = ['actinic_keratosis', 'basal_cell_carcinoma', 'dermatofibroma', 'melanoma']
SIZE = 224  # ResNet50 input size

def getData():
    rawdata = []
    data = []
    dir = "C:/Users/RETECH-01/Desktop/data/data/"
    for category in categories:
        path = os.path.join(dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                rawdata = cv2.imread(os.path.join(path, img))
                new_data = cv2.resize(rawdata, (SIZE, SIZE))

                data.append([new_data, class_num])
            except Exception as e:
                pass

    random.shuffle(data)

    img_data = []
    img_labels = []
    for features, label in data:
        img_data.append(features)
        img_labels.append(label)
    img_data = np.array(img_data)
    img_labels = np.array(img_labels)

    return img_data, img_labels

data, labels = getData()
data = data.astype('float32') / 255.0  # Normalize pixel values

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, labels, test_size=0.20)
train_data, val_data, train_labels, val_labels = model_selection.train_test_split(train_data, train_labels, test_size=0.10)
print(len(train_data), " ", len(train_labels), len(test_data), " ", len(test_labels))

# Load pre-trained ResNet50 model without the top (classification) layer
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

# Freeze the layers in the pre-trained model
for layer in resnet_model.layers:
    layer.trainable = False

# Add custom classification layers
flatten_layer = layers.Flatten()(resnet_model.output)
output_layer = layers.Dense(len(categories), activation='softmax')(flatten_layer)

# Create the model
model = keras.models.Model(inputs=resnet_model.input, outputs=output_layer)

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_data, train_labels, epochs=100, validation_data=(val_data, val_labels))

# Save the model
model.save('./model/ResNet50_skin.h5')
