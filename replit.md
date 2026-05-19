# VideoGuard — AI Video Tampering Detector

A web app that uses a trained deep learning model to detect tampering in uploaded videos. Users upload a video, the model analyzes sampled frames, and returns a verdict (Authentic / Tampered) with a confidence score and per-frame breakdown.

## Run & Operate

- **Flask app**: `python artifacts/flask-app/app.py` (port 5000)
- **Workflow**: `Start application` — runs the Flask backend

## Stack

- **Backend**: Python + Flask
- **ML**: TensorFlow / Keras (`tamper_model.h5`)
- **Video**: OpenCV (`opencv-python-headless`)
- **Frontend**: Jinja2 templates + Tailwind CSS CDN + Vanilla JS (single-page, no build step)

## Where things live

- `artifacts/flask-app/app.py` — Flask routes and model inference logic
- `artifacts/flask-app/tamper_model.h5` — Trained Keras model (13MB)
- `artifacts/flask-app/templates/index.html` — Full single-page UI
- `artifacts/flask-app/uploads/` — Temp upload folder (files deleted after analysis)
- `artifacts/flask-app/requirements.txt` — Python dependencies

## Architecture decisions

- Every 10th frame is sampled from the video and resized to 128×128 before inference — matches the original training setup.
- Uploaded files are deleted immediately after analysis to save disk space.
- Model is loaded lazily on first request to avoid slow server startup.
- JSON API at `/api/predict` keeps the frontend decoupled from Flask templating.
- `opencv-python-headless` is used instead of `opencv-python` since there's no display server.

## Product

Users upload a video (MP4, AVI, MOV, MKV, WebM — up to 500MB). The app extracts frames, runs each through the tampering detection model, and returns:
- **Verdict**: Authentic or Tampered
- **Confidence**: percentage certainty
- **Frame chart**: per-frame tampering probability bar chart

## Gotchas

- The model expects 128×128 RGB frames normalized to [0, 1].
- TensorFlow emits INFO logs to stderr on startup — these are harmless.
- Do NOT run `pnpm run dev` at workspace root — it has no dev script.
- Always check that the `Start application` workflow is running before testing.
