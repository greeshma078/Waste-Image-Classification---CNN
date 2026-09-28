import streamlit as st
from PIL import Image

from src.predict import load_model, predict_image


st.set_page_config(
    page_title="Waste Image Classification",
    page_icon="♻️",
    layout="centered"
)


st.title("♻️ Waste Image Classification")

st.write(
    "Upload an image and the CNN model will "
    "classify it into one of six waste categories."
)


@st.cache_resource
def get_model():
    return load_model()


model = get_model()


uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Classify Image"):

        predicted_class, confidence = predict_image(
            model,
            image
        )

        st.success(
            f"Prediction: {predicted_class.title()}"
        )

        st.info(
            f"Confidence: {confidence * 100:.2f}%"
        )


st.markdown("---")

st.subheader("Supported Waste Categories")

st.write(
    """
    ♻️ Cardboard

    🍾 Glass

    🔩 Metal

    📄 Paper

    🧴 Plastic

    🗑️ Trash
    """
)