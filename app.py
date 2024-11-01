# app.py
import streamlit as st
import cv2
import numpy as np
import utils
import swensens_menu

st.set_page_config(page_title="Menu Recommendation", layout="centered")
st.title("Swensen's Menu Recommendation Kiosk 🍽️")
st.write("Capture an image to get menu recommendations based on your detected attributes.")

# Camera Input Widget
image_data = st.camera_input("Capture an image")

if image_data is not None:
    # Convert image data to a format OpenCV can process
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Predict image attributes
    visualized_image, detected_emotion, detected_age, detected_gender = utils.predict_image(frame)

    # Display image with annotations
    st.image(visualized_image, channels="BGR")
    st.write(f"Emotion: {detected_emotion}")
    st.write(f"Age: {detected_age}")
    st.write(f"Gender: {detected_gender}")

    # Get menu recommendations based on detected attributes
    menu_recommendations = swensens_menu.recommend_menu(detected_emotion, detected_age, detected_gender)

    # Display final concise recommendation
    st.write("Here’s a personalized food recommendation based on your profile:")
    st.write(f"Recommended Items: {', '.join(menu_recommendations)}")
