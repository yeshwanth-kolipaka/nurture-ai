# Nurture AI

A multimodal infant cry detection web app that analyzes baby videos using **audio + visual deep learning** to classify cries and emotional states.

## How it works

1. Upload a video of your baby
2. Audio is extracted via FFmpeg and processed through a cry classification pipeline (MFCC features + LSTM-based models)
3. Video frames are analyzed using an EfficientNet-based emotion recognition model
4. Results are fused using weighted multimodal reasoning to produce a final state with recommendations (e.g. hungry, tired, distressed)

## Features

- User authentication (signup/login with hashed passwords)
- Video upload and analysis
- Audio pipeline: binary cry detection → multiclass cry classification (belly_pain, discomfort, hungry, tired, burping)
- Visual pipeline: frame extraction → EfficientNet-based emotion classification (distressed, sleepy, normal)
- Weighted multimodal fusion with semantic reasoning
- Analysis history per user
- Session-based result caching

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Database | MongoDB (via PyMongo) |
| Audio ML | TensorFlow / Keras + Librosa |
| Visual ML | EfficientNet (via TensorFlow) |
| Video Processing | OpenCV |
| Audio Extraction | FFmpeg |
| Auth | Werkzeug (password hashing + sessions) |

## Models

- `models/binary_cry_model.keras` — binary cry/no-cry classifier
- `models/multiclass_cry_model.keras` — multiclass cry type classifier
- `models/visual_emotion_model.keras` — visual emotion classifier (EfficientNet)
- `models/*.npy` — class names and normalization statistics

## Setup

1. Clone the repo
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file (see `.env.example`):

```
MONGO_URI=mongodb://your-mongo-uri
FLASK_SECRET_KEY=your-secret-key
```

4. Run the app:

```bash
python app.py
```

## Requirements

See `requirements.txt` — Flask, TensorFlow, OpenCV, NumPy, Librosa, PyMongo, python-dotenv.

## License

MIT
