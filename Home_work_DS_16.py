import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt

model_cnn = load_model("fashion_mnist_model_keras.h5")
model_vgg16 = load_model("fashion_mnist_model_VGG16.h5")
with open("history.pkl", "rb") as file:
    history_kernes = pickle.load(file)
with open("history_2.pkl", "rb") as file:
    history_vgg16 = pickle.load(file)

labels = [
    "T-shirt/top",
    "Trousers",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def plot_metrics(history_dict):
    st.subheader("Графік точності")
    plt.figure(figsize=(10, 4))
    plt.plot(history_dict["accuracy"], label="Точність тренування")
    plt.plot(history_dict["val_accuracy"], label="Валідаційна точність")
    plt.title("Точність моделі")
    plt.xlabel("Епохи")
    plt.ylabel("Точність")
    plt.legend()
    st.pyplot(plt)

    st.subheader("Графік втрат")
    plt.figure(figsize=(10, 4))
    plt.plot(history_dict["loss"], label="Втрата тренування")
    plt.plot(history_dict["val_loss"], label="Валідаційні втрати")
    plt.title("Втрати моделі")
    plt.xlabel("Епохи")
    plt.ylabel("Втрати")
    plt.legend()
    st.pyplot(plt)


def preprocess_image(image, model_type):
    if model_type == "Згорткова нейромережа":
        img = image.convert("L")
        img = img.resize((28, 28))
        img = np.array(img).astype("float32") / 255
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
    else:
        img = image.convert("L")
        img = img.resize((32, 32))
        img = np.array(img).astype("float32") / 255
        img = np.expand_dims(img, axis=-1)
        img = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img))
        img = np.expand_dims(img, axis=0)
    return img


st.title("Класифікація зображень")
model_choice = st.selectbox(
    "Оберіть модель:", ("Згорткова нейромережа", "Модель на основі VGG16")
)

uploaded_file = st.file_uploader("Завантажте зображення", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Завантажене зображення", use_column_width=True)
    img = preprocess_image(image, model_choice)

    if model_choice == "Згорткова нейромережа":
        predictions = model_cnn.predict(img)
        plot_metrics(history_kernes)
    else:
        predictions = model_vgg16.predict(img)
        plot_metrics(history_vgg16)

    class_index = np.argmax(predictions)
    predicted_label = labels[class_index]
    predicted_probabilities = predictions[0]

    st.write("Ймовірності для кожного класу:")
    for i, label in enumerate(labels):
        st.write(f"{label}: {predicted_probabilities[i]:.2f}")

    st.write(f"Клас передбачення: **{predicted_label}**")
