import os
import json
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 128
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}

model = None


def load_model():
    global model
    if model is None:
        import tensorflow as tf
        model_path = os.path.join(os.path.dirname(__file__), 'tamper_model.h5')
        model = tf.keras.models.load_model(model_path)
    return model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    import cv2

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Please upload MP4, AVI, MOV, MKV, or WebM.'}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(filepath)
        keras_model = load_model()

        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        predictions = []
        frame_count = 0
        analyzed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:
                resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                normalized = resized / 255.0
                expanded = np.expand_dims(normalized, axis=0)
                prediction = keras_model.predict(expanded, verbose=0)[0][0]
                predictions.append(float(prediction))
                analyzed_frames += 1

            frame_count += 1

        cap.release()

        if not predictions:
            return jsonify({'error': 'Could not read video frames. Please try a different video.'}), 400

        average_prediction = float(np.mean(predictions))
        tampered = average_prediction > 0.5
        confidence = average_prediction if tampered else (1 - average_prediction)

        frame_analysis = []
        for i, p in enumerate(predictions):
            frame_analysis.append({
                'frame': i * 10,
                'score': round(float(p), 4),
                'tampered': p > 0.5
            })

        result = {
            'verdict': 'Tampered' if tampered else 'Authentic',
            'tampered': tampered,
            'confidence': round(confidence * 100, 1),
            'raw_score': round(average_prediction, 4),
            'total_frames': total_frames,
            'analyzed_frames': analyzed_frames,
            'duration_seconds': round(duration, 2),
            'frame_analysis': frame_analysis[:50],
            'filename': file.filename,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
