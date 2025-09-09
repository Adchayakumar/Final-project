
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------- helper: centre-crop digit ----------
def centre_crop_digit(img_arr, pad=10):
    """
    img_arr: 0-255 grayscale uint8 numpy array
    returns: 28×28 grayscale float32 ready for MNIST model
    """
    # Otsu threshold
    _, th = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find bounding box
    coords = cv2.findNonZero(th)
    if coords is None:              # blank image
        return np.zeros((28, 28), dtype=np.float32)
    x, y, w, h = cv2.boundingRect(coords)

    # add padding
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img_arr.shape[1] - x, w + 2*pad)
    h = min(img_arr.shape[0] - y, h + 2*pad)

    digit = img_arr[y:y+h, x:x+w]

    # resize while keeping aspect ratio
    if h > w:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, int(h * 20 / w))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # paste onto 28×28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    xoff = (28 - new_w) // 2
    yoff = (28 - new_h) // 2
    canvas[yoff:yoff+new_h, xoff:xoff+new_w] = digit

    # scale to 0-1 and return
    return canvas.astype(np.float32) / 255.0


# ---------- UI ----------
st.title("MNIST Digit Recogniser")
st.write("Upload any image that contains a single handwritten digit (0-9).")

uploaded = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("L")          # grayscale
    img_np  = np.array(pil_img)

    # preprocess
    processed = centre_crop_digit(img_np)
    processed_batch = processed.reshape(1, 28, 28, 1)    # add batch & channel dims

    # load model (cached)
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("/content/digit_cnn.h5")
    model = load_model()

    # predict
    probs = model.predict(processed_batch, verbose=0)[0]
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])

    # show results
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_column_width=True)
    with col2:
        st.image(processed, caption="28×28 pre-processed", clamp=True, use_column_width=True)

    st.success(f"Predicted digit: **{pred}**  (confidence {conf*100:.1f}%)")
    st.bar_chart(probs)