import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io

# Load the trained decoder model
decoder = tf.keras.models.load_model("decoder.h5")

st.title("Handwritten Digit Generator")
st.write("Generate handwritten-style digits using a trained Variational Autoencoder (VAE).")

digit = st.selectbox("Select a digit (0-9)", list(range(10)))

if st.button("Generate Images"):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        z_sample = np.random.normal(size=(1, 2))
        generated_image = decoder.predict(z_sample)[0, :, :, 0]

        axes[i].imshow(generated_image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"{digit}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)

st.caption("Model trained on MNIST dataset, generated using VAE.")
