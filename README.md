# EmotionLens 🎭

![EmotionLens](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688)

EmotionLens is a real-time, multi-person facial emotion recognition web application. Built upon a custom-trained **EfficientNet-B0** model and powered by **YuNet** for robust face detection, it captures and analyzes facial expressions in real-time. 

## 🌟 Key Features
- **🚀 Real-time Processing**: Smooth webcam streaming (~12fps) via WebSockets.
- **👥 Multi-Face Tracking**: Accurate multi-person face detection powered by OpenCV's YuNet.
- **🧠 7-Class Emotion Engine**: Robust EfficientNet-B0 model trained for real-world faces, classifying 7 distinct emotions (`neutral`, `happiness`, `surprise`, `sadness`, `anger`, `disgust`, `fear`).
- **🎨 5 Interactive Lenses (Modes)**: Built-in modes for different scenarios, such as Canteen Satisfaction, Sudden Emotion Alert, Movie Reviewer, Speech Coach, and an interactive Mimic Game.

---

## 🏗️ Design Philosophy: "One Engine, Multiple Lenses"

Our core design philosophy is to decouple the **computation-heavy Emotion Engine** from the **lightweight Application Lenses**.

```text
[ Emotion Engine ] ──► (Real-time Emotion Stream) ──► [ Lens 1: Speech Coach ]
                                                  ──► [ Lens 2: Sudden Alert ]
                                                  ──► [ Lens N: Custom Lens  ]
```

1. **The Engine (Backend)**: Takes a webcam frame, runs face detection once, runs the EfficientNet model once, and broadcasts a structured JSON containing bounding boxes and emotion probabilities.
2. **The Lenses (Frontend/Aggregators)**: Consume the exact same emotion stream but interpret it differently. A "Speech Coach" lens looks for confidence (happiness/neutral), while a "Sudden Alert" lens triggers only on sudden spikes of fear or anger. 

This means **you don't need to retrain the model or touch the AI pipeline** to build a brand new application scenario. You just snap on a new "Lens".

---

## 🚀 How to Run

### 1. Install Requirements
Ensure you have Python 3.8+ installed. Then install the dependencies:
```bash
pip install -r requirements.txt
```
*(Make sure you have OpenCV >= 4.7.0 for YuNet support.)*

### 2. Start the Server
You can start the FastAPI backend server using the provided batch script or directly via python:
```bash
# On Windows
cd emotion-app
start.bat

# Or manually:
python main.py
```

### 3. Open the App
Open your browser and navigate to:
```
http://localhost:8000
```
Grant webcam permissions, and the system will start streaming and analyzing emotions in real-time. Swipe or click to switch between different lenses at the bottom of the screen.

---

## 🛠️ How to Add Your Own "Lens"

Because of the "One Engine, Multiple Lenses" architecture, extending the project with a new lens is incredibly straightforward. You can create a new Lens either entirely on the Frontend, or as a Backend module.

### Approach: Adding a Frontend-only Lens
The backend continuously streams emotion data via WebSockets. You can simply catch this data and draw a new widget.

1. **Listen to the WebSocket data**:
   In `emotion-app/frontend/app.js`, the `ws.onmessage` function receives data formatted like this:
   ```json
   {
     "faces": [
       {
         "box": [x, y, w, h],
         "dominant": "happiness",
         "conf": 0.95,
         "probs": {"happiness": 0.95, "neutral": 0.02, ...}
       }
     ],
     "fps": 12.5
   }
   ```
2. **Create a new HTML section**:
   Add a new `<div class="swiper-slide">` in `index.html` for your lens UI.
3. **Consume the data**:
   Write a JavaScript function that reads the `faces` array and updates your UI. For example, if you are making a "Classroom Focus Tracker", you might calculate the ratio of `neutral` and `happiness` vs `sadness` and `anger` in the `faces` array, and update a progress bar.

### Backend-driven Lenses
If your lens requires complex state management (like saving historical data to a database, or triggering an email alert), you can add a new python module in `emotion-app/backend/lenses/`.
1. Create a new file `my_custom_lens.py`.
2. Write an aggregator class that processes the raw inference output.
3. Hook it into the WebSocket dispatcher in `emotion-app/backend/modes.py` to push processed data to the frontend.

Enjoy building on top of EmotionLens!
