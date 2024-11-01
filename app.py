import streamlit as st
import PIL
import numpy as np
import io
import tempfile
import utils

def play_video(video_source):
    import cv2
    camera = cv2.VideoCapture(video_source)
    fps = camera.get(cv2.CAP_PROP_FPS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_row = []

    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0
    st_frame = st.empty()

    while camera.isOpened():
        ret, frame = camera.read()

        if ret:
            visualized_image, detected_emotion, detected_age, detected_gender = utils.predict_image(frame)
            st_frame.image(visualized_image, channels="BGR")
            st.write(f"Detected Emotion: {detected_emotion}, Age: {detected_age}, Gender: {detected_gender}")
            video_row.append(cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB))

            frame_count += 1
            progress_bar.progress(frame_count / total_frames)

        else:
            camera.release()
            st_frame.empty()
            progress_bar.empty()
            break

    # Save the video output
    clip = utils.create_video_clip(video_row, fps)
    st.video(temp_file.name)

st.set_page_config(
    page_title="AI Fire Safety Project",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("AI Smoke and Fire Detection 🔥")

st.sidebar.header("Input Source")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 0, 100, 20)) / 100

if source_radio == "IMAGE":
    st.sidebar.header("Upload Image")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = np.array(uploaded_image)
        visualized_image, detected_emotion, detected_age, detected_gender = utils.predict_image(uploaded_image_cv)
        st.image(visualized_image, channels="BGR")
        st.write(f"Detected Emotion: {detected_emotion}, Age: {detected_age}, Gender: {detected_gender}")

elif source_radio == "VIDEO":
    st.sidebar.header("Upload Video")
    input = st.sidebar.file_uploader("Choose a video.", type=("mp4"))

    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4"
        
        with open(temporary_location, "wb") as out:
            out.write(g.read())

        play_video(temporary_location)

elif source_radio == "WEBCAM":
    play_video(0)
