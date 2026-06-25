<div align="center">

# 🍼 Nurture AI
### Multimodal Infant Cry Detection & Emotion Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.3-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

---

> 🎯 **Audio + Visual fusion** · Real-time baby cry classification · Weighted multimodal reasoning

---

## What It Does

Nurture AI analyzes uploaded baby videos through a **multimodal deep learning pipeline** — combining **audio cry classification** (MFCC + LSTM) with **visual emotion recognition** (EfficientNet) to detect what your baby needs.

**Audio classes:** belly_pain, discomfort, hungry, tired, burping, no_cry  
**Visual classes:** distressed, sleepy, normal

---

## Multimodal Pipeline

```
Video Upload
    │
    ├── Audio Stream ──→ FFmpeg Extraction ──→ MFCC + Delta Features
    │                                               │
    │                                    Binary Cry Classifier (no_cry / cry)
    │                                               │
    │                                    if cry → Multiclass Cry Classifier
    │                                               │
    │                                    {belly_pain, discomfort, hungry, tired, burping}
    │
    ├── Video Stream ──→ Frame Sampling (20 frames) ──→ EfficientNet Emotion Model
    │                                                       │
    │                                    {distressed, sleepy, normal, ...}
    │
    ▼
Weighted Multimodal Fusion (audio_weight=0.75, video_weight=0.25)
    │
    ▼
Semantic Reasoning → Final State + Recommendation
```

### Fusion Logic

| Condition | Final State | Recommendation |
|---|---|---|
| belly_pain/discomfort + distressed + confidence ≥ 0.65 | "Baby appears in significant distress" | Immediate caregiver attention |
| hungry detected | "Baby appears hungry" | Feeding may help |
| tired or sleepy | "Baby appears sleepy" | Rest or sleep may be needed |
| burping detected | "Baby may need burping" | Gentle burping posture |
| normal + confidence ≥ 0.70 | "Baby appears calm and stable" | No immediate concern |

---

## Key Features

- 🎥 **Video upload & analysis** — MP4, MOV, AVI, MKV, WebM supported (200MB max)
- 🔊 **Binary cry detection** — gates multiclass model to save compute
- 🎯 **5-class cry classification** — belly_pain, discomfort, hungry, tired, burping
- 👶 **Visual emotion recognition** — EfficientNet-based frame-by-frame analysis
- 🧠 **Weighted multimodal fusion** — audio 0.75 / video 0.25 with semantic reasoning
- 👤 **User auth** — signup/login with hashed passwords (Werkzeug)
- 📜 **Analysis history** — per-user result storage in MongoDB
- 📄 **Session caching** — last result available without re-upload

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask |
| Database | MongoDB (via PyMongo) |
| Deep learning | TensorFlow / Keras |
| Audio processing | Librosa (MFCC, delta features) |
| Vision model | EfficientNet (pre-trained) |
| Video processing | OpenCV |
| Audio extraction | FFmpeg |
| Auth | Werkzeug (password hashing + sessions) |

---

## Model Architecture

| Model | Task | Input | Output |
|---|---|---|---|
| `binary_cry_model.keras` | Cry vs no-cry | MFCC features | Binary probability |
| `multiclass_cry_model.keras` | Cry type classification | MFCC features | 5-class probabilities |
| `visual_emotion_model.keras` | Visual emotion | 240×240 RGB frames | Emotion class |

All models are loaded at startup from the `models/` directory.

---

## Local Setup

### Prerequisites
- Python 3.10+
- MongoDB instance (local or Atlas)
- FFmpeg installed and in PATH

### 1. Clone
```bash
git clone https://github.com/yeshwanth-kolipaka/nurture-ai.git
cd nurture-ai
```

### 2. Virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment variables
```bash
cp .env.example .env
# Fill in MONGO_URI and FLASK_SECRET_KEY in .env
```

### 5. Run
```bash
python app.py
# → http://127.0.0.1:5000
```

---

## Requirements

```
flask
tensorflow
opencv-python
numpy
librosa
soundfile
imageio
imageio-ffmpeg
pymongo
python-dotenv
```

---

## Project Structure

```
nurture-ai/
├── app.py                       # Flask application (routes, auth, upload)
├── db.py                        # MongoDB connection & collections
├── requirements.txt
├── .env.example
├── .gitignore
├── fusion/
│   └── prediction_pipeline.py   # Multimodal pipeline (audio + video)
├── models/
│   ├── binary_cry_model.keras
│   ├── multiclass_cry_model.keras
│   ├── visual_emotion_model.keras
│   └── *.npy                    # Class names & normalization stats
├── static/
│   ├── css/style.css
│   └── js/script.js
├── templates/
│   ├── base.html
│   ├── homepage.html
│   ├── login.html
│   ├── signup.html
│   ├── result.html
│   ├── history.html
│   └── feedback.html
└── uploads/                     # Uploaded videos (gitignored)
```

---

## Contact

**Yeshwanth** — [yeshwanthkolipaka1505@gmail.com](mailto:yeshwanthkolipaka1505@gmail.com)

🔗 [GitHub Repo](https://github.com/yeshwanth-kolipaka/nurture-ai)

---

<div align="center">
  <sub>Built with ❤️ for babies and caregivers</sub>
</div>
