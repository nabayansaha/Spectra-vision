import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
# from model import model


def build_model():
    return keras.Sequential([
        layers.Rescaling(1./255, input_shape=(128, 128, 3)),

        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # binary classification
    ])
