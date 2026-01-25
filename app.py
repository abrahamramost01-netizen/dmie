import os
import uuid
import json
import psycopg2
import cv2
import numpy as np
import onnxruntime as ort

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.environ.get("MODEL_PATH", "model.onnx")
WIN_POINTS = int(os.environ.get("WIN_POINTS", 200))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# ================= DATABASE =================
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= LOAD ONNX MODEL =================
print(f"🔧 Cargando modelo ONNX: {MODEL_PATH}")

try:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print("✅ Modelo ONNX cargado correctamente")
except Exception as e:
    print("❌ Error cargando ONNX:", e)
    session = None

# ================= UTILS =================
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2 - x1) * (y2 - y1) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / union if union > 0 else 0

def nms(detections, iou_thresh=0.5):
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    final = []

    while detections:
        best = detections.pop(0)
        final.append(best)
        detections = [
            d for d in detections
            if iou(best["box"], d["box"]) < iou_thresh
        ]
    return final

# ================= DETECTION =================
def calcular_puntos_domino(image_path):
    if session is None:
        return {"total": 0, "cantidad": 0, "fichas": [], "error": "Modelo no cargado"}

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    blob = cv2.resize(img, (640, 640))
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    outputs = session.run(None, {input_name: blob})[0]

    detections = []

    for det in outputs:
        conf = float(det[4])
        if conf < 0.5:
            continue

        cls = int(np.argmax(det[5:]))
        x, y, bw, bh = det[:4]

        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        detections.append({
            "box": (x1, y1, x2, y2),
            "class": cls,
            "conf": conf
        })

    detections = nms(detections)

    total = 0
    fichas = []

    for d in detections:
        puntos = d["class"]  # domino doble-9 → clase = puntos
        total += puntos

        fichas.append({
            "puntos": puntos,
            "confianza": round(d["conf"] * 100, 1)
        })

        cv2.rectangle(img, d["box"][:2], d["box"][2:], (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{puntos}",
            (d["box"][0], d["box"][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    out_path = image_path.replace(".", "_det.")
    cv2.imwrite(out_path, img)

    return {
        "total": total,
        "cantidad": len(fichas),
        "fichas": fichas,
        "image_debug": out_path
    }

# ================= ROUTES =================
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
    team_id = int(request.form["team_id"])
    image = request.files["image"]

    filename = f"{uuid.uuid4()}.jpg"
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
        resultado.get("image_debug", path),
        json.dumps(resultado)
    ))

    cur.execute("UPDATE teams SET points = points + %s WHERE id = %s",
                (resultado["total"], team_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= START =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

