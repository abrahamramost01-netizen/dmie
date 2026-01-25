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
def detect():import os
import uuid
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "best.onnx"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cargar modelo SOLO una vez
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


@app.route("/")
def index():
    return render_template("test.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_path": MODEL_PATH,
        "onnx_loaded": True
    })


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img = cv2.imread(path)
    h, w = img.shape[:2]

    # Preprocesado YOLO típico
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    outputs = session.run(None, {input_name: img_input})[0]

    detections = []

    for det in outputs[0]:
        conf = float(det[4])
        if conf < 0.5:
            continue

        cls = int(det[5])

        # ⚠️ NO reescalar mal, estos valores YA están en píxeles
        x1, y1, x2, y2 = map(int, det[:4])

        detections.append({
            "box": [x1, y1, x2, y2],
            "clase": cls,
            "conf": conf
        })

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out_name = f"proc_{filename}"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    cv2.imwrite(out_path, img)

    return jsonify({
        "detections": detections,
        "image": out_name
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)




