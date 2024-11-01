import openvino as ov
import cv2
import numpy as np

# Load the OpenVINO core
core = ov.Core()

# Load and compile face detection model
model_face = core.read_model(model='model/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model=model_face, device_name="CPU")

# Get input and output layers for face detection
input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

# Load and compile emotion recognition model
model_emo = core.read_model(model='model/emotions-recognition-retail-0003.xml')
compiled_model_emo = core.compile_model(model=model_emo, device_name="CPU")

# Get input and output layers for emotion recognition
input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

# Load and compile age-gender recognition model
model_ag = core.read_model(model='model/age-gender-recognition-retail-0013.xml')
compiled_model_ag = core.compile_model(model=model_ag, device_name="CPU")

# Get input and output layers for age-gender recognition
input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output(0)

def preprocess(image, input_layer):
    """Preprocess image for model input."""
    _, input_channels, input_height, input_width = input_layer.shape
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)  # Channel first
    input_image = np.expand_dims(transposed_image, 0)
    return input_image

def find_faceboxes(image, results):
    """Find face bounding boxes."""
    results = results.squeeze()
    scores = results[:, 2]
    boxes = results[:, -4:]
    face_boxes = boxes[scores >= 0.5]  # Use a fixed threshold
    image_h, image_w, _ = image.shape
    face_boxes = face_boxes * np.array([image_w, image_h, image_w, image_h])
    face_boxes = face_boxes.astype(np.int64)
    return face_boxes

def draw_age_gender_emotion(face_boxes, image):
    """Draw boxes and add age, gender, emotion labels."""
    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    show_image = image.copy()
    detected_emotion = "neutral"
    detected_age = 0
    detected_gender = "unknown"

    for i, box in enumerate(face_boxes):
        xmin, ymin, xmax, ymax = box
        face = image[ymin:ymax, xmin:xmax]

        # --- Emotion detection ---
        input_image = preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_image])[output_layer_emo].squeeze()
        detected_emotion = EMOTION_NAMES[np.argmax(results_emo)]

        # --- Age and gender detection ---
        input_image_ag = preprocess(face, input_layer_ag)
        results_ag = compiled_model_ag([input_image_ag])
        detected_age = int(results_ag[1].squeeze() * 100)
        gender = results_ag[0].squeeze()
        detected_gender = 'female' if gender[0] >= 0.65 else 'male' if gender[1] >= 0.55 else "unknown"

        # Draw the detections
        cv2.rectangle(show_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{detected_gender} {detected_age} {detected_emotion}"
        cv2.putText(show_image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return show_image, detected_emotion, detected_age, detected_gender

def predict_image(image):
    """Main function to detect face, age, gender, and emotion."""
    input_image = preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes = find_faceboxes(image, results)
    return draw_age_gender_emotion(face_boxes, image)