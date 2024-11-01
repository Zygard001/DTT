# app.py
import cv2
import numpy as np
import utils
import swensens_menu
import time
import streamlit as st

def play_video(video_source=0, max_frames=10):
    # Check camera access
    camera = cv2.VideoCapture(video_source)

    if not camera.isOpened():
        st.error("Error: Camera not accessible.")
        return

    # Camera initialization
    st.write("Initializing the camera...")
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)  # Simulate setup time
        progress_bar.progress(percent_complete + 1)

    st.write("Camera initialized. Please position your face in front of the camera...")

    st_frame = st.empty()
    frame_skip = 10  # Process every 10th frame for reduced processing frequency
    frame_count = 0
    all_recommendations = []

    while camera.isOpened() and frame_count < max_frames:
        ret, frame = camera.read()
        if ret:
            if frame_count % frame_skip == 0:
                # Detect attributes
                visualized_image, detected_emotion, detected_age, detected_gender = utils.predict_image(frame)

                # Display image and results temporarily
                st_frame.image(visualized_image, channels="BGR")
                st.write(f"Emotion: {detected_emotion}")
                st.write(f"Age: {detected_age}")
                st.write(f"Gender: {detected_gender}")

                # Collect recommendations for the final summary
                menu_recommendations = swensens_menu.recommend_menu(detected_emotion, detected_age, detected_gender)
                all_recommendations.extend(menu_recommendations)

            frame_count += 1
        else:
            st.warning("No frame captured from camera.")
            break

    camera.release()

    # Consolidate recommendations with deduplication
    unique_recommendations = list(set(all_recommendations))  # Remove duplicates

    # Final display with clear categorization
    st.write("### Recommended Items Based on Analysis:")
    st.write(f"Detected Emotion: {detected_emotion}")
    st.write("Recommended Items:")
    st.write(", ".join(unique_recommendations))

st.set_page_config(page_title="Menu Recommendation", layout="centered")
st.title("Swensen's Menu Recommendation Kiosk 🍽️")
st.write("Processing video stream...")

if st.button('Start Video'):
    play_video(0)