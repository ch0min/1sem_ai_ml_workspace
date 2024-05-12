import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import pickle

DIRECTORY = "./data/"
CATEGORIES = ["cats", "dogs"]


# Specifying the directory path to the data images as well as resizing:
IMG_SIZE = 100
data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)

    # Index of labels
    label = CATEGORIES.index(category)

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)

        # 100 x 100
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))

        # Image with corresponding label to indicate whether it's a cat or dog.
        data.append([img_arr, label])

        # plt.imshow(img_arr)
        # break


len(data)

random.shuffle(data)

X = []
y = []

for features, labels in data:
    X.append(features)
    y.append(labels)

# Converting to numpy:
X = np.array(X)
y = np.array(y)

# Saving data:
pickle.dump(X, open("X.pkl", "wb"))
pickle.dump(y, open("y.pkl", "wb"))

### Training the Data (Could change to a new notebook)

import pickle

X = pickle.load(open("X.pkl", "rb"))
y = pickle.load(open("y.pkl", "rb"))

# Feature scaling since RGB is 0-255:
X = X / 255

# Now it's 0 to 1
X

# Img count, width, height, channels:
X.shape

# Logging
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = f"cats_vs_dogs_prediction_{int(time.time())}"

tensorboard = TensorBoard(log_dir=f"logs\\{NAME}\\")


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, input_shape=X.shape[1:], activation="relu"))
model.add(Dense(128, activation="relu"))


model.add(Dense(2, activation="softmax"))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(X, y, epochs=5, validation_split=0.1, batch_size=32, callbacks=[tensorboard])


# (be inside directory OLA-4) tensorboard --logdir=logs/
