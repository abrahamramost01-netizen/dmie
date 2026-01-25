import os
import uuid
import json
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template, send_from_directory

# ================= CONFIG =================
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
IMG_SIZE = 640
CONF_THRESHOLD = 0.01  # BAJO para debug
IOU_THRESHOLD = 0.45

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)

# ================= LOAD MODEL =================
try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_loaded = True
except Exception as e:
    print("❌ Error ONNX:", e)
    session = None
    onnx_loaded = False

# ================= UTILS =================
def preprocess(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    img_r = img_r.astype(np.float32) / 255.0
    img_r = np.transpose(img_r, (2, 0, 1))
    img_r = np.expand_dims(img_r, axis=0)
    return img, img_r, w, h

# ================= INFERENCE =================
def infer_raw(image_path):
    img_orig, img_in, w, h = preprocess(image_path)

    outputs = session.run([output_name], {input_name: img_in})[0]

    detections = []

    for det in outputs[0]:
        conf = float(det[4])
        cls = int(det[5])

        if conf < CONF_THRESHOLD:
            continue

        cx, cy, bw, bh = det[0:4]

        # OJO: esto explica tus números gigantes
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        detections.append({
            "box": [x1, y1, x2, y2],
            "clase": cls,
            "conf": conf * 100
        })

        cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            img_orig,
            f"{cls} {conf*100:.1f}%",
            (x1, max(20, y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    name = f"proc_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, name), img_orig)

    return {
        "detections": detections,
        "image": name
    }

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("test.html")

@app.route("/infer", methods=["POST"])
def infer():
    if not onnx_loaded:
        return jsonify({"error": "ONNX no cargado"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    img = request.files["image"]
    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    img.save(path)

    result = infer_raw(path)
    return jsonify(result)

@app.route("/processed/<name>")
def processed(name):
    return send_from_directory(PROCESSED_FOLDER, name)

@app.route("/health")
def health():
    return {
        "status": "ok",
        "onnx_loaded": onnx_loaded,
        "model_path": MODEL_PATH
    }

# ================= START =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


