import cv2
from PIL import Image
import streamlit as st
from helper import detect_plate, read_plate

DETECTION_MODEL_PATH = "./models/plate_detection.pt"
READ_MODEL_PATH = "./models/plate_reading.pt"

# UI
st.title("Plate Recognition System 🚗")

st.header("Upload an Image")
file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# Image show
if file is not None:
    st.header("Image")
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Detect plates
    st.header("Plate Detection")

    # Detection function
    detected_image, cropped_images, is_detected = detect_plate(image, DETECTION_MODEL_PATH)

    if is_detected != 0:
        st.success(f"Plates detected: {is_detected}")
        
        # Show the image with bounding boxes for detected plates
        st.subheader("Detected Plates")
        st.image(detected_image, caption="Detected Plates", use_container_width=True)

        # Show cropped plates separately
        st.subheader("Cropped Plates")
        for idx, cropped_img in enumerate(cropped_images):
            st.image(cropped_img, use_container_width=True)

            # Read and display text from each cropped plate
            st.subheader(f"Read Plate {idx+1} Text")
            read_image, plate_text = read_plate(cropped_img, READ_MODEL_PATH)
            st.image(read_image, use_container_width=True)
            st.markdown(f"<h2 style='color: green;'>Detected Plate Text: {plate_text}</h2>", unsafe_allow_html=True)

    else:
        st.error("No plate detected.")
        st.image(detected_image, caption="Processed Image", use_container_width=True)
