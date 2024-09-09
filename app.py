import keras
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

if 'predicted' not in st.session_state:
    st.session_state.predicted = 0


@st.cache_resource
def get_model():
    return keras.models.load_model("../handwritten1.keras")


canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=25,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=224,
    key="canvas",
    width=224
)

btn = st.button("Predict")
if btn:
    img = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
    img = img.resize((28, 28))
    # img.save("test.png", format="PNG")
    data = np.invert([np.array(img)[:, :, 0]])
    data = keras.utils.normalize(data, axis=1)
    prediction = get_model().predict(data)
    st.session_state.predicted = np.argmax(prediction)

st.text(st.session_state.predicted)
