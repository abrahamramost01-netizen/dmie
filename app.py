import os
import uuid
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, render_template, send_from_directory, jsonify

UPLOADS = "uploads"
PROCESSED = "processed"
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")

os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)

app = Flask(__name__)

# ===== LOAD MODEL =====
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("test.html")

@app.route("/detect", methods=["POST"])
def detect():
    img_file = request.files["image"]
    name = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOADS, name)
    img_file.save(path)

    img = cv2.imread(path)
    h, w = img.shape[:2]

    inp = cv2.resize(img, (640, 640))
    inp = inp[:, :, ::-1].astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))
    inp = np.expand_dims(inp, 0)

    outputs = session.run([output_name], {input_name: inp})[0]

    detections = []

    for det in outputs[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        detections.append({
            "box": [x1, y1, x2, y2],
            "clase": int(cls),
            "conf": conf
        })

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            img,
            f"{int(cls)} ({conf:.2f})",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    out_name = f"proc_{name}"
    cv2.imwrite(os.path.join(PROCESSED, out_name), img)

    return jsonify({
        "detections": detections,
        "image": out_name
    })

@app.route("/processed/<name>")
def processed(name):
    return send_from_directory(PROCESSED, name)

@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "onnx_loaded": True
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



