import tensorflow as tf

# 0.9514, 0.9522, 0.9514
# OPTIMIZER: AdaMax
# EPOCHS: 20

TEST_SIZE = 0.2
IMG_WIDTH = 128
IMG_HEIGHT = 128
EPOCHS = 20

def get_model():
	model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(1, activation="sigmoid")
        ])

	model.compile(
	    optimizer="AdaMax",
	    loss="binary_crossentropy",
	    metrics=["accuracy"]
	    )

	return model
