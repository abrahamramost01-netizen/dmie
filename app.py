import os
import uuid
import json
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, render_template, jsonify, send_from_directory

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)

# ================= LOAD MODEL =================
session = None
input_name = None
output_name = None
model_error = None

try:
    if os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
    else:
        model_error = f"Model not found: {MODEL_PATH}"
except Exception as e:
    model_error = str(e)

# ================= UTILS =================
def preprocess(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    img_r = img_r.astype(np.float32) / 255.0
    img_r = np.transpose(img_r, (2, 0, 1))
    img_r = np.expand_dims(img_r, axis=0)
    return img, img_r, w, h

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (a[2]-a[0]) * (a[3]-a[1])
    area2 = (b[2]-b[0]) * (b[3]-b[1])
    union = area1 + area2 - inter
    return inter / union if union else 0

def nms(dets):
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    out = []
    for d in dets:
        if all(iou(d["box"], o["box"]) < IOU_THRESHOLD for o in out):
            out.append(d)
    return out

# ================= CORE =================
def infer_image(path, draw=False):
    if session is None:
        return {"error": model_error}

    img, tensor, w, h = preprocess(path)
    raw = session.run([output_name], {input_name: tensor})

    detections = []
    for d in raw[0][0]:
        conf = float(d[4])
        if conf < CONF_THRESHOLD:
            continue
        cls = int(d[5])
        cx, cy, bw, bh = d[:4]
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        detections.append({
            "clase": cls,
            "conf": conf,
            "box": [x1, y1, x2, y2]
        })

    detections = nms(detections)

    if draw:
        for d in detections:
            cv2.rectangle(img, (d["box"][0], d["box"][1]),
                          (d["box"][2], d["box"][3]), (0,255,0), 2)
            cv2.putText(img, f'{d["clase"]} {int(d["conf"]*100)}%',
                        (d["box"][0], d["box"][1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out_name = f"proc_{os.path.basename(path)}"
        cv2.imwrite(os.path.join(PROCESSED_FOLDER, out_name), img)
        return {"detections": detections, "image": out_name}

    return {"detections": detections, "raw_output": raw}

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/infer", methods=["POST"])
def infer():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no image"}), 400

    name = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_FOLDER, name)
    f.save(path)

    result = infer_image(path, draw=True)
    return jsonify(result)

@app.route("/infer_raw", methods=["POST"])
def infer_raw():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no image"}), 400

    name = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_FOLDER, name)
    f.save(path)

    return jsonify(infer_image(path, draw=False))

@app.route("/processed/<name>")
def processed(name):
    return send_from_directory(PROCESSED_FOLDER, name)

@app.route("/debug")
def debug():
    return jsonify({
        "status": "ok",
        "onnx_loaded": session is not None,
        "model_path": MODEL_PATH,
        "error": model_error
    })

# ================= START =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
