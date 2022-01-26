import tensorflow as tf

# NUMBER 1
# 0.9573, 0.9488, 0.9462, 0.9565, 0.9428 (0.95032)
# EPOCHS: 10

# NUMBER 2 (MOST OPTIMAL)
# 0.9599, 0.9471, 0.9420, 0.9514, 0.9462 (0.94932)
# EPOCHS: 20

# NUMBER 3

# different loss functions

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
	    optimizer="adam",
	    loss="binary_crossentropy",
	    metrics=["accuracy"]
	    )

	return model
