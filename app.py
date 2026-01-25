import os
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

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
MAX_FICHAS = 55  # Dominó doble-9

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ================== DATABASE ==================
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================== LOAD ONNX ==================
print(f"🔧 Cargando modelo ONNX: {MODEL_PATH}")
try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("✅ Modelo ONNX cargado")
except Exception as e:
    print(f"❌ Error ONNX: {e}")
    session = None

# ================== UTILS ==================
def preprocess(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = np.transpose(img_norm, (2, 0, 1))
    img_norm = np.expand_dims(img_norm, axis=0)
    return img, img_norm, w, h

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def non_max_suppression(detections):
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    final = []
    for det in detections:
        keep = True
        for f in final:
            if iou(det["box"], f["box"]) > IOU_THRESHOLD:
                keep = False
                break
        if keep:
            final.append(det)
    return final[:MAX_FICHAS]

# ================== CORE ==================
def calcular_puntos_domino(image_path):
    if session is None:
        return {"total": 0, "cantidad": 0, "fichas": [], "error": "Modelo no cargado"}

    img_original, img_input, w, h = preprocess(image_path)
    outputs = session.run([output_name], {input_name: img_input})[0]

    detections = []

    for det in outputs[0]:
        conf = float(det[4])
        cls = int(det[5])
        if conf < CONF_THRESHOLD:
            continue

        cx, cy, bw, bh = det[0:4]
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

    detections = non_max_suppression(detections)

    fichas = []
    total = 0

    for d in detections:
        total += d["puntos"]
        fichas.append({
            "clase": d["clase"],
            "puntos": d["puntos"],
            "confianza": round(d["conf"] * 100, 1)
        })

        # dibujar caja
        cv2.rectangle(
            img_original,
            (d["box"][0], d["box"][1]),
            (d["box"][2], d["box"][3]),
            (0, 255, 0),
            2
        )
        cv2.putText(
            img_original,
            f"{d['puntos']} ({int(d['conf']*100)}%)",
            (d["box"][0], d["box"][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    processed_name = f"proc_{os.path.basename(image_path)}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_name)
    cv2.imwrite(processed_path, img_original)

    return {
        "total": total,
        "cantidad": len(fichas),
        "fichas": fichas,
        "imagen_procesada": processed_name
    }

# ================== ROUTES ==================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points, m.image_path, m.details
        FROM matches m
        JOIN teams t ON t.id = m.team_id
        ORDER BY m.id DESC
    """)
    matches = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("index.html", teams=teams, matches=matches, win_points=WIN_POINTS)

@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = int(request.form.get("team_id"))
    image = request.files.get("image")

    if not image:
        return redirect(url_for("index"))

    ext = os.path.splitext(image.filename)[1].lower()
    filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    resultado = calcular_puntos_domino(path)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO matches (team_id, points, image_path, details)
        VALUES (%s, %s, %s, %s)
    """, (
        team_id,
        resultado["total"],
        resultado.get("imagen_procesada"),
        json.dumps(resultado)
    ))

    cur.execute(
        "UPDATE teams SET points = points + %s WHERE id = %s",
        (resultado["total"], team_id)
    )

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/processed/<filename>")
def processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route("/health")
def health():
    return {"status": "ok", "onnx": session is not None}

# ================== START ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))




