import os
import uuid
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PATH = "best.onnx"

CONF_THRESHOLD = 100  # IMPORTANTE: tu modelo NO usa 0–1

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ================== LOAD MODEL ==================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ================== PREPROCESS ==================
def preprocess(path):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = np.transpose(img_norm, (2, 0, 1))
    img_norm = np.expand_dims(img_norm, axis=0)
    return img, img_norm

# ================== INFERENCE ==================
def infer_raw(path):
    img, img_input = preprocess(path)
    outputs = session.run([output_name], {input_name: img_input})[0]

    detections = []

    for det in outputs[0]:
        conf = float(det[4])
        cls = int(det[5])

        if conf < CONF_THRESHOLD:
            continue

        # 🚨 YA VIENEN EN PIXELES
        x1, y1, x2, y2 = map(int, det[:4])

        detections.append({
            "box": [x1, y1, x2, y2],
            "clase": cls,
            "conf": conf
        })

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            img,
            f"{cls} ({int(conf)})",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    out_name = f"proc_{os.path.basename(path)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, out_name), img)

    return detections, out_name

# ================== ROUTES ==================
@app.route("/", methods=["GET", "POST"])
def debug():
    if request.method == "POST":
        img = request.files["image"]
        name = f"{uuid.uuid4()}.jpg"
        path = os.path.join(UPLOAD_FOLDER, name)
        img.save(path)

        detections, out_img = infer_raw(path)

        return jsonify({
            "detections": detections,
            "image": out_img
        })

    return render_template("debug.html")

@app.route("/processed/<name>")
def processed(name):
    return app.send_static_file(f"../processed/{name}")

@app.route("/health")
def health():
    return {
        "status": "ok",
        "onnx_loaded": True,
        "model_path": MODEL_PATH
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



