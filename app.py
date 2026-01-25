import os
import uuid
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template, send_from_directory

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")

IMG_SIZE = 640
CONF_THRESHOLD = 0.5

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ================== LOAD MODEL ==================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ================== UTILS ==================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(path):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = np.transpose(img_norm, (2, 0, 1))
    img_norm = np.expand_dims(img_norm, axis=0)
    return img, img_norm

# ================== CORE ==================
def run_raw_yolo(image_path):
    img, inp = preprocess(image_path)
    output = session.run([output_name], {input_name: inp})[0]

    detections = []

    for det in output[0]:
        raw_conf = det[4]
        conf = sigmoid(raw_conf)

        if conf < CONF_THRESHOLD:
            continue

        cls = int(det[5])

        # ⚠️ NO escalar cajas
        x1, y1, x2, y2 = map(int, det[0:4])

        detections.append({
            "box": [x1, y1, x2, y2],
            "clase": cls,
            "conf": float(conf)
        })

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            img,
            f"{cls} {conf:.2f}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    name = f"proc_{uuid.uuid4()}.jpg"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, name), img)

    return {
        "detections": detections,
        "image": name
    }

# ================== ROUTES ==================
@app.route("/")
def index():
    return render_template("test.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("image")
    name = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_FOLDER, name)
    file.save(path)

    result = run_raw_yolo(path)
    return jsonify(result)

@app.route("/processed/<name>")
def processed(name):
    return send_from_directory(PROCESSED_FOLDER, name)

@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "onnx_loaded": True
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



