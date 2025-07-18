# Maths_AI_problem_solver
✋🧮 Hand Gesture Math Solver with Gemini AI This project is an interactive math problem solver that combines:  Hand gesture detection using OpenCV &amp; cvzone,  Image-based equation recognition with Google Gemini AI,  Streamlit web interface for real-time use.
🔍 Features
✍️ Write math problems with hand gestures (index finger on webcam canvas)

🖼️ Upload images of equations to get solutions instantly

🤖 AI-powered solving using Google Gemini (generative AI)

🖐️ Gesture commands:

1 finger = draw

5 fingers = clear canvas

4 fingers = process input

🛠️ Tech Stack
Python

OpenCV

Streamlit

cvzone (Hand Tracking Module)

Google Generative AI (Gemini)

🚀 How to Run
Clone this repo

Install dependencies (pip install -r requirements.txt)

Add your Gemini API key in the code (genai.configure)

Run with:

bash
Copy
Edit
streamlit run test4.py
📌 Notes
Make sure your webcam is enabled.

Include a hand_gestures.png guide image in the root directory.

AI requires internet access and a valid API key.
