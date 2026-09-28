import numpy as np
import tensorflow as tf


MODEL_PATH = "models/waste_cnn_final.keras"

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def predict_image(model, image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[predicted_index]

    return predicted_class, confidence