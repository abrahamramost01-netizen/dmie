import os
import uuid
import json
import psycopg2
import cv2import os
import uuid
import json
import psycopg2
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# ================== CONFIG ==================
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")

WIN_POINTS = int(os.environ.get("WIN_POINTS", 200))
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
MAX_FICHAS = 55  # dominó doble-9

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ================== DATABASE ==================
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================== LOAD ONNX ==================
print(f"🔧 Cargando modelo ONNX: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("✅ Modelo ONNX listo")

# ================== UTILS ==================
def preprocess(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return img, np.expand_dims(x, 0), w, h

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def nms(dets):
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    out = []
    for d in dets:
        if all(iou(d["box"], o["box"]) < IOU_THRESHOLD for o in out):
            out.append(d)
        if len(out) >= MAX_FICHAS:
            break
    return out

# ================== CORE ==================
def calcular_puntos_domino(image_path):
    img, x, w, h = preprocess(image_path)
    preds = session.run([output_name], {input_name: x})[0][0]

    detections = []

    for p in preds.T:
        obj_conf = p[4]
        if obj_conf < CONF_THRESHOLD:
            continue

        class_probs = p[5:]
        cls = int(np.argmax(class_probs))
        cls_conf = class_probs[cls]
        conf = obj_conf * cls_conf

        if cls > 9:
            continue  # dominó doble-9

        cx, cy, bw, bh = p[:4]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        detections.append({
            "box": [x1, y1, x2, y2],
            "clase": cls,
            "puntos": cls,
            "conf": conf
        })

    detections = nms(detections)

    total = 0
    fichas = []

    for d in detections:
        total += d["puntos"]
        fichas.append({
            "clase": d["clase"],
            "puntos": d["puntos"],
            "confianza": round(d["conf"] * 100, 1)
        })

        cv2.rectangle(img, (d["box"][0], d["box"][1]),
                             (d["box"][2], d["box"][3]), (0,255,0), 2)

        cv2.putText(img,
            f"{d['puntos']} ({int(d['conf']*100)}%)",
            (d["box"][0], d["box"][1]-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out_name = f"proc_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, out_name), img)

    return {
        "total": total,
        "cantidad": len(fichas),
        "fichas": fichas,
        "imagen_procesada": out_name
    }

# ================== ROUTES ==================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/add_match", methods=["POST"])
def add_match():
    image = request.files.get("image")
    if not image:
        return redirect(url_for("index"))

    ext = os.path.splitext(image.filename)[1]
    name = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_FOLDER, name)
    image.save(path)

    result = calcular_puntos_domino(path)
    return render_template("result.html", result=result, img=name)

@app.route("/uploads/<f>")
def uploads(f):
    return send_from_directory(UPLOAD_FOLDER, f)

@app.route("/processed/<f>")
def processed(f):
    return send_from_directory(PROCESSED_FOLDER, f)

@app.route("/health")
def health():
    return {"status": "ok", "onnx": True}

# ================== START ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))






