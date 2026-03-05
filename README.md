# Real-Time Facial Emotion Recognition (FER)

A real-time computer vision system that detects faces via webcam
and classifies emotions live using deep learning.

## Emotions Detected
angry, disgust, fear, happy, neutral, sad, surprise

## Tech Stack
- Python 3.10
- OpenCV (face detection + video capture)
- FER library (TensorFlow/Keras CNN model)
- NumPy (temporal smoothing)
- Matplotlib + Pandas (analytics dashboard)

## Project Structure
```
fer_project/
    main.py              # entry point
    utils/
        overlay.py       # drawing functions
        logger.py        # CSV + JSON logging
    logs/                # saved session data
    main.ipynb           # development notebook
```

## How to Run
```
# Activate environment
conda activate tf_env

# Run live detector
python main.py
```

## Features
- Live webcam emotion detection at 20+ FPS
- Colored bounding box per emotion
- Confidence bar + all 7 emotion scores overlay
- Session dominant emotion tracker
- CSV logging of every prediction
- Post-session analytics dashboard (5 charts)

## Results
Achieves 20+ FPS on CPU using:
- Frame skipping (detect every 2nd frame)
- Frame scaling (60% size for detection)
- Temporal smoothing (5-frame average)