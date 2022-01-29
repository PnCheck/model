import os
import sys
import importlib
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import model

# Labels
# Normal: 0
# Pneumonia: 1

DATA_DIR = "chest_xray"
FILENAME = "model.h5"

# Load the data
images = []
labels = []

i = 0

for category in os.listdir(DATA_DIR):
	directory = os.path.join(DATA_DIR, category)

	for file in os.listdir(directory):
		image = cv2.imread(os.path.join(directory, file), cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (model.IMG_WIDTH, model.IMG_HEIGHT))

		images.append(image)
		labels.append(1 if category == "pneumonia" else 0)

		i += 1
		print(i)

x_train, x_test, y_train, y_test = train_test_split(
    	np.array(images), np.array(labels), test_size=model.TEST_SIZE
    	)

def train():
	importlib.reload(model)

	new_model = model.get_model()

	new_model.fit(x_train, y_train, epochs=model.EPOCHS)

	new_model.evaluate(x_test,  y_test, verbose=2)

	new_model.save(FILENAME)
	
	print(f"Model saved to {FILENAME}.")
