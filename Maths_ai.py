import numpy as np
import cv2
from PIL import Image
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai

# Initialize Google Generative AI configuration
genai.configure(api_key="AIzaSyBnthAKUtNqe815Y-hD-j4oJvA34u2TE8k")
model = genai.GenerativeModel("gemini-1.5-flash")

# Set up Streamlit page configuration
st.set_page_config(layout='wide')
st.title("Math Problem Solver: Hand Gestures & File Upload")
st.write("""
This project combines hand gesture detection and file upload to solve math problems using Google Gemini AI.
\n**Features**:
- Draw math problems using your finger and control with hand gestures.
- Upload a math problem image, and get the solution instantly.
\n**Instructions**:
1. **Hand Gestures**:
    - Raise **4 fingers** to process the equation.
    - Raise **5 fingers** to erase the canvas.
    - Use **1 finger** to write the equation.
2. **File Upload**:
    - Upload an image of a math equation, and it will be solved automatically.
""")

# Display PNG guide image (optional)
gesture_image = Image.open("hand_gestures.png")  # Ensure the image file is in the folder
st.image(gesture_image, caption="Hand Gesture Guide", use_column_width=True)

# Answer display section
st.title("Answer")
answer_placeholder = st.empty()

# File uploader for math problem image
uploaded_file = st.file_uploader("Upload an Image with a Math Problem", type=["jpg", "jpeg", "png"])

# Configure columns for gesture detection and file upload
column1, column2 = st.columns([2, 1])

# Hand gesture detection configurations
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

canvas = None
previous_position = None
output_result = ""

# Helper function to send image or canvas to Generative AI
def process_with_ai(image):
    try:
        # Save the image locally for AI processing
        image_pil = Image.fromarray(image)
        image_pil.save("temp_image.jpg")
        with open("temp_image.jpg", "rb") as img_file:
            response = model.generate_content(["solve this math problem"], img_file.read())
        return response.text if response else "Error: No response from Gemini"
    except Exception as e:
        return f"Error: {e}"

# Gesture-based drawing function
def draw_gesture(info, canvas, previous_position):
    fingers, lm_list = info
    current_position = None

    if fingers == [0, 1, 0, 0, 0]:  # One finger up (write)
        current_position = lm_list[8][0:2]  # Get index finger position
        if previous_position is not None:
            cv2.line(canvas, previous_position, current_position, (255, 0, 0), 10)
        previous_position = current_position

    elif fingers == [1, 1, 1, 1, 1]:  # Five fingers up (erase canvas)
        canvas = np.zeros_like(canvas)

    elif fingers == [1, 1, 1, 1, 0]:  # Four fingers up (process equation)
        return process_with_ai(canvas), canvas

    return None, canvas

# Process uploaded image
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_np = np.array(image)  # Convert PIL image to numpy array
    output_result = process_with_ai(image_np)
    styled_answer = f'<div style="border: 2px solid black; padding: 10px; border-radius: 5px; background-color: #f9f9f9; color: #000;">{output_result}</div>'
    answer_placeholder.markdown(styled_answer, unsafe_allow_html=True)

# Hand Gesture Detection (Webcam)
with column1:
    frame_window = st.image(np.zeros((720, 1280, 3), dtype=np.uint8), channels="BGR")

if cap.isOpened():
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Flip the image for mirroring
        if canvas is None:
            canvas = np.zeros_like(img)

        hands, img = detector.findHands(img, flipType=False, draw=False)
        if hands:
            hand_info = detector.fingersUp(hands[0]), hands[0]["lmList"]
            result, canvas = draw_gesture(hand_info, canvas, previous_position)
            if result:
                styled_answer = f'<div style="border: 2px solid black; padding: 10px; border-radius: 5px; background-color: #f9f9f9; color: #000;">{result}</div>'
                answer_placeholder.markdown(styled_answer, unsafe_allow_html=True)

        # Combine the original image and the canvas
        combined_image = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        frame_window.image(combined_image, channels="BGR")

        if cv2.waitKey(1) == ord('q'):  # Exit on pressing 'q'
            break

cap.release()
cv2.destroyAllWindows()
