import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

(train_images, train_labels), (test_images, test_labels) = (
    datasets.fashion_mnist.load_data()
)

train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

train_images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(train_images))
test_images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(test_images))

train_images = tf.image.resize(train_images, (32, 32))
test_images = tf.image.resize(test_images, (32, 32))

train_images = train_images.numpy().astype("float32") / 255.0
test_images = test_images.numpy().astype("float32") / 255.0

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
validation_generator = test_datagen.flow(test_images, test_labels, batch_size=32)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

for layer in base_model.layers[:-4]:
    layer.trainable = False

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_images) // 32,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=len(test_images) // 32,
)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

model.save("fashion_mnist_model_VGG16.h5")

with open("history_2.pkl", "wb") as file:
    pickle.dump(history.history, file)
