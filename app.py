import os
import uuid
import json
import psycopg2
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
WIN_POINTS = int(os.environ.get("WIN_POINTS", 200))
MODEL_PATH = os.environ.get("MODEL_PATH", "model.onnx")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)

# ================= LOAD MODEL =================
print(f"🔧 Cargando modelo ONNX: {MODEL_PATH}")
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    model_loaded = True
    print("✅ Modelo ONNX cargado")
except Exception as e:
    print("❌ Error cargando modelo:", e)
    net = None
    model_loaded = False

# ================= DATABASE =================
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= UTILIDADES =================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union else 0

def fusionar_cajas(detecciones, iou_thr=0.45):
    detecciones = sorted(detecciones, key=lambda x: x["conf"], reverse=True)
    final = []
    for d in detecciones:
        if all(iou(d["box"], f["box"]) < iou_thr for f in final):
            final.append(d)
    return final

# ================= YOLO DOMINÓ =================
def calcular_puntos_domino(image_path):
    if not model_loaded:
        return {"total": 0, "cantidad": 0, "fichas": [], "error": "Modelo no cargado"}

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img, 1/255.0, (640, 640), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward()[0]

    detecciones = []

    for det in outputs:
        conf = det[4]
        if conf < 0.5:
            continue

        class_scores = det[5:]
        cls = int(np.argmax(class_scores))
        score = class_scores[cls] * conf
        if score < 0.5:
            continue

        cx, cy, bw, bh = det[:4]
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        detecciones.append({
            "clase": cls,
            "puntos": cls,
            "conf": float(score),
            "box": [x1, y1, x2, y2]
        })

    # 🔒 FUSIÓN DE CAJAS
    detecciones = fusionar_cajas(detecciones)

    # 🎯 DOBLE NUEVE: máximo 55 fichas
    detecciones = detecciones[:55]

    total = sum(d["puntos"] for d in detecciones)

    # 🖌️ DIBUJAR CAJAS
    out = img.copy()
    for d in detecciones:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            out,
            f"{d['puntos']} ({int(d['conf']*100)}%)",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    result_img = f"{uuid.uuid4()}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_img)
    cv2.imwrite(result_path, out)

    return {
        "total": total,
        "cantidad": len(detecciones),
        "imagen_resultado": result_img,
        "fichas": [
            {
                "clase": d["clase"],
                "puntos": d["puntos"],
                "confianza": round(d["conf"]*100, 1)
            } for d in detecciones
        ]
    }

# ================= ROUTES =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()
    cur.execute("""
        SELECT m.id, t.name, m.points, m.details
        FROM matches m JOIN teams t ON t.id=m.team_id
        ORDER BY m.id DESC
    """)
    matches = cur.fetchall()
    cur.close(); conn.close()
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
    cur.execute(
        "INSERT INTO matches (team_id, points, details) VALUES (%s,%s,%s)",
        (team_id, resultado["total"], json.dumps(resultado))
    )
    cur.execute("UPDATE teams SET points = points + %s WHERE id=%s",
                (resultado["total"], team_id))
    conn.commit()
    cur.close(); conn.close()

    return redirect(url_for("index"))

@app.route("/results/<filename>")
def results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

