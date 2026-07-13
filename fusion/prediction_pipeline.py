# =========================================================
# MULTIMODAL INFANT ANALYSIS PIPELINE
# =========================================================

# =========================================================
# IMPORTS
# =========================================================

import os
import cv2
import librosa
import numpy as np
import subprocess
import zipfile
import json
import tempfile
import atexit

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.efficientnet import (
    preprocess_input
)

_TEMP_FILES = []


@atexit.register
def _cleanup():
    for f in _TEMP_FILES:
        try:
            os.unlink(f)
        except OSError:
            pass


def _strip_qc(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            _strip_qc(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_qc(item)


def _load_cleaned_model(path):
    tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    tmp.close()
    _TEMP_FILES.append(tmp.name)
    with zipfile.ZipFile(path, "r") as zin:
        with zipfile.ZipFile(tmp.name, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "config.json":
                    config = json.loads(data)
                    _strip_qc(config)
                    data = json.dumps(config).encode()
                zout.writestr(item, data)
    return load_model(tmp.name)

# =========================================================
# CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# MULTIMODAL WEIGHTS
# ---------------------------------------------------------

AUDIO_WEIGHT = 0.75

VIDEO_WEIGHT = 0.25

# ---------------------------------------------------------
# VIDEO CONFIG
# ---------------------------------------------------------

IMAGE_SIZE = 240

MAX_VIDEO_FRAMES = 20

VIDEO_ANALYSIS_DURATION = 10

# ---------------------------------------------------------
# AUDIO CONFIG
# ---------------------------------------------------------

SAMPLE_RATE = 16000

DURATION = 5

N_MFCC = 40

MAX_TIME_STEPS = 100

# =========================================================
# LOAD VIDEO MODEL
# =========================================================

visual_model = _load_cleaned_model(
    "models/visual_emotion_model.keras"
)

VISUAL_CLASSES = list(

    np.load(

        "models/visual_class_names.npy",

        allow_pickle=True
    )
)

# load the binary model

binary_model = _load_cleaned_model(
    "models/binary_cry_model.keras"
)

binary_mean = np.load(
    "models/mean.npy"
)

binary_std = np.load(
    "models/std.npy"
)

# =========================================================
# LOAD AUDIO MODEL
# =========================================================

audio_model = _load_cleaned_model(
    "models/multiclass_cry_model.keras"
)

audio_mean = np.load(
    "models/audio_mean.npy"
)

audio_std = np.load(
    "models/audio_std.npy"
)

AUDIO_CLASSES = list(

    np.load(

        "models/audio_class_names.npy",

        allow_pickle=True
    )
)

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_multimodal_pipeline(video_path):

    # =====================================================
    # CHECK VIDEO
    # =====================================================

    if not os.path.exists(video_path):

        return {

            "error": "Video not found"
        }

    # =====================================================
    # VIDEO PROCESSING
    # =====================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        return {

            "error": "Could not open video"
        }

    frames = []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps and fps > 0:

        analysis_frames = min(
            total_frames,
            int(fps * VIDEO_ANALYSIS_DURATION)
        )

    else:

        analysis_frames = total_frames

    if analysis_frames > 0:

        frame_positions = np.linspace(
            0,
            max(analysis_frames - 1, 0),
            num=min(MAX_VIDEO_FRAMES, analysis_frames),
            dtype=np.int32
        )

    else:

        frame_positions = np.arange(
            MAX_VIDEO_FRAMES,
            dtype=np.int32
        )

    for frame_position in frame_positions:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_position)
        )

        ret, frame = cap.read()

        if not ret:
            continue

        try:

            # ---------------------------------------------
            # BGR → RGB
            # ---------------------------------------------

            frame = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB
            )

            # ---------------------------------------------
            # RESIZE
            # ---------------------------------------------

            frame = cv2.resize(

                frame,

                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )

            # ---------------------------------------------
            # FLOAT ARRAY
            # ---------------------------------------------

            frame = np.array(

                frame,

                dtype=np.float32
            )

            # ---------------------------------------------
            # EFFICIENTNET PREPROCESS
            # ---------------------------------------------

            frame = preprocess_input(frame)

            frames.append(frame)

        except Exception as e:

            print(
                f"Frame Error: {e}"
            )

            continue

    cap.release()

    # =====================================================
    # CHECK FRAMES
    # =====================================================

    if len(frames) == 0:

        return {

            "error": "No valid frames extracted"
        }

    # =====================================================
    # VIDEO PREDICTION
    # =====================================================

    frames = np.array(frames)

    video_predictions = visual_model.predict(

        frames,

        verbose=0
    )

    # =====================================================
    # FRAME AGGREGATION
    # =====================================================

    video_probabilities = np.mean(

        video_predictions,

        axis=0
    )

    video_index = np.argmax(
        video_probabilities
    )

    video_result = VISUAL_CLASSES[
        video_index
    ]

    video_confidence = float(

        video_probabilities[video_index]
    )

    # =====================================================
    # DEFAULT AUDIO RESULT
    # =====================================================

    audio_result = "no_audio"

    audio_confidence = 0.0
    audio_probabilities = None
    binary_result = "unknown"
    binary_confidence = 0.0

    # =====================================================
    # AUDIO PROCESSING
    # =====================================================

    try:

        # -------------------------------------------------
        # TEMP AUDIO PATH (unique per request)
        # -------------------------------------------------

        audio_tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        )
        audio_output_path = audio_tmp.name
        audio_tmp.close()
        _TEMP_FILES.append(audio_output_path)

        # -------------------------------------------------
        # FFmpeg COMMAND
        # -------------------------------------------------

        ffmpeg_command = [

            "ffmpeg",

            "-loglevel", "error",

            "-i", video_path,

            "-t", str(DURATION),

            "-vn",

            "-acodec", "pcm_s16le",

            "-ar", "16000",

            "-ac", "1",

            audio_output_path,

            "-y"
        ]

        # -------------------------------------------------
        # RUN FFmpeg
        # -------------------------------------------------

        subprocess.run(

            ffmpeg_command,

            check=True,

            stdout=subprocess.DEVNULL,

            stderr=subprocess.DEVNULL
        )

        # -------------------------------------------------
        # CHECK AUDIO FILE
        # -------------------------------------------------

        if os.path.exists(audio_output_path):

            # =============================================
            # LOAD AUDIO
            # =============================================

            audio, sr = librosa.load(

                audio_output_path,

                sr=SAMPLE_RATE
            )

            # =============================================
            # FIX AUDIO LENGTH
            # =============================================

            required_length = (
                SAMPLE_RATE * DURATION
            )

            if len(audio) < required_length:

                padding = (

                    required_length
                    -
                    len(audio)
                )

                audio = np.pad(

                    audio,

                    (0, padding)
                )

            else:

                audio = audio[
                    :required_length
                ]

            # =============================================
            # NORMALIZE (for binary model — matches its training)
            # =============================================

            audio = librosa.util.normalize(audio)

            # =============================================
            # BINARY FEATURES (no preemphasis)
            # =============================================

            bin_mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=N_MFCC
            )
            bin_delta = librosa.feature.delta(bin_mfcc)
            bin_delta2 = librosa.feature.delta(bin_mfcc, order=2)
            bin_features = np.concatenate(
                [bin_mfcc, bin_delta, bin_delta2], axis=0
            ).T

            if bin_features.shape[0] < MAX_TIME_STEPS:
                pad = MAX_TIME_STEPS - bin_features.shape[0]
                bin_features = np.pad(
                    bin_features, ((0, pad), (0, 0))
                )
            else:
                bin_features = bin_features[:MAX_TIME_STEPS]

            binary_features = (
                (bin_features - binary_mean) / (binary_std + 1e-6)
            )

            binary_input = np.expand_dims(
                binary_features, axis=0
            )

            # =============================================
            # PRE-EMPHASIS (for multiclass model — matches its training)
            # =============================================

            audio = librosa.effects.preemphasis(audio)

            # =============================================
            # MULTICLASS FEATURES (with preemphasis)
            # =============================================

            mc_mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=N_MFCC
            )
            mc_delta = librosa.feature.delta(mc_mfcc)
            mc_delta2 = librosa.feature.delta(mc_mfcc, order=2)
            mc_features = np.concatenate(
                [mc_mfcc, mc_delta, mc_delta2], axis=0
            ).T

            if mc_features.shape[0] < MAX_TIME_STEPS:
                pad = MAX_TIME_STEPS - mc_features.shape[0]
                mc_features = np.pad(
                    mc_features, ((0, pad), (0, 0))
                )
            else:
                mc_features = mc_features[:MAX_TIME_STEPS]

            multiclass_features = (
                (mc_features - audio_mean) / (audio_std + 1e-6)
            )

            # =============================================
            # BINARY PREDICTION
            # =============================================

            binary_prob = binary_model.predict(

                binary_input,

                verbose=0

            )[0][0]

            binary_confidence = float(
                binary_prob
            )

            # =============================================
            # NO CRY
            # =============================================

            if binary_prob < 0.5:

                binary_result = "no_cry"

                audio_result = "no_cry"

                audio_confidence = float(
                    1 - binary_prob
                )

            # =============================================
            # CRY DETECTED
            # =============================================

            else:

                binary_result = "cry"

                audio_input = np.expand_dims(

                    multiclass_features,

                    axis=0
                )

                audio_prediction = audio_model.predict(

                    audio_input,

                    verbose=0
                )[0]

                audio_probabilities = audio_prediction

                audio_index = np.argmax(
                    audio_prediction
                )

                audio_result = AUDIO_CLASSES[
                    audio_index
                ]

                audio_confidence = float(
                    audio_prediction[audio_index]
                )

    except Exception as e:

        print("\nAudio Processing Failed")

        print(e)

    # =====================================================
    # WEIGHTED FUSION
    # =====================================================

    if audio_result in ("no_cry", "no_audio"):

        current_audio_weight = 0.0

        current_video_weight = 1.0

    else:

        current_audio_weight = AUDIO_WEIGHT

        current_video_weight = VIDEO_WEIGHT

    weighted_audio_score = (

        audio_confidence * current_audio_weight
    )

    weighted_video_score = (

        video_confidence * current_video_weight
    )

    # =====================================================
    # FINAL CONFIDENCE
    # =====================================================

    final_confidence = (

        weighted_audio_score

        +

        weighted_video_score
    )

    # =====================================================
    # SEMANTIC MULTIMODAL REASONING
    # =====================================================

    final_state = ""

    reasoning = ""

    recommendation = ""

    # =====================================================
    # CASE 1 — HIGH DISTRESS
    # =====================================================

    if (

        audio_result in [

            "belly_pain",

            "discomfort"
        ]

        and

        video_result == "distressed"

        and

        final_confidence >= 0.65
    ):

        final_state = (

            "Baby appears to be in "
            "significant distress"
        )

        reasoning = (

            "Pain/discomfort cry pattern "
            "combined with distressed "
            "visual expression"
        )

        recommendation = (

            "Immediate caregiver attention "
            "recommended"
        )

    # =====================================================
    # CASE 2 — HUNGRY
    # =====================================================

    elif (

        audio_result == "hungry"
    ):

        final_state = (

            "Baby appears hungry"
        )

        reasoning = (

            "Hungry cry pattern "
            "strongly detected"
        )

        recommendation = (

            "Feeding may help calm the baby"
        )

    # =====================================================
    # CASE 3 — TIRED / SLEEPY (both modalities must agree)
    # =====================================================

    elif (

        audio_result == "tired"

        and

        video_result == "sleepy"
    ):

        final_state = (

            "Baby appears sleepy or tired"
        )

        reasoning = (

            "Both audio and visual cues "
            "indicate tiredness"
        )

        recommendation = (

            "Rest or sleep may be needed"
        )

    # =====================================================
    # CASE 4 — BURPING
    # =====================================================

    elif (

        audio_result == "burping"
    ):

        final_state = (

            "Baby may need burping"
        )

        reasoning = (

            "Burping-related cry "
            "pattern detected"
        )

        recommendation = (

            "Gentle burping posture "
            "may help"
        )

    # =====================================================
    # CASE 5 — NO CRY DETECTED
    # =====================================================

    elif (

        audio_result == "no_cry"
    ):

        final_state = (

            "No crying detected"
        )

        reasoning = (

            f"Audio: no_cry "
            f"({round(audio_confidence, 2)}), "
            f"Video: {video_result} "
            f"({round(video_confidence, 2)})"
        )

        recommendation = (

            "Baby appears calm — "
            "no immediate concern"
        )

    # =====================================================
    # CASE 6 — NORMAL (visual only)
    # =====================================================

    elif (

        video_result == "normal"

        and

        video_confidence >= 0.70
    ):

        final_state = (

            "Baby appears calm and stable"
        )

        reasoning = (

            "Normal emotional state "
            "detected visually"
        )

        recommendation = (

            "No immediate concern detected"
        )

    # =====================================================
    # DEFAULT CASE
    # =====================================================

    else:

        final_state = (

            "Unable to determine baby's "
            "state with high confidence"
        )

        reasoning = (

            f"Audio: {audio_result} "
            f"({round(audio_confidence, 2)}), "
            f"Video: {video_result} "
            f"({round(video_confidence, 2)})"
        )

        recommendation = (

            "Continue monitoring "
            "baby behavior"
        )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    return {
        "binary_result": binary_result,

        "binary_confidence": round(
            binary_confidence,
            4
        ),

        "audio_result": audio_result,

        "audio_confidence": round(
            audio_confidence,
            4
        ),

        "video_result": video_result,

        "video_confidence": round(
            video_confidence,
            4
        ),

        "final_state": final_state,

        "final_confidence": round(
            final_confidence,
            4
        ),

        "reasoning": reasoning,

        "recommendation": recommendation
    }
