import os
import uuid
import json
import psycopg2
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO

# ================== CONFIG ==================
UPLOAD_FOLDER = "uploads"
PRED_FOLDER = "uploads/pred"
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")
WIN_POINTS = int(os.environ.get("WIN_POINTS", 200))

CONF_THRES = 0.45
IOU_THRES = 0.5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)

app = Flask(__name__)

# ================== CARGAR MODELO ==================
print(f"🔧 Cargando modelo ONNX: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Modelo ONNX cargado correctamente")
except Exception as e:
    print("❌ ERROR cargando modelo:", e)
    model = None


# ================== DB ==================
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


# ================== UTILIDADES ==================
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


def fusionar_cajas(detecciones):
    detecciones = sorted(detecciones, key=lambda x: x["conf"], reverse=True)
    resultado = []

    for det in detecciones:
        solapa = False
        for r in resultado:
            if iou(det["box"], r["box"]) > IOU_THRES:
                solapa = True
                break
        if not solapa:
            resultado.append(det)

    return resultado


# ================== YOLO DOMINÓ ==================
def calcular_puntos_domino(image_path):
    if model is None:
        return {"total": 0, "cantidad": 0, "fichas": [], "error": "Modelo no cargado"}

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    results = model(image_path, conf=CONF_THRES, verbose=False)

    detecciones = []

    for r in results:
        for b in r.boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            detecciones.append({
                "clase": cls,
                "puntos": cls,  # DOBLE NUEVE → clase = puntos
                "conf": conf,
                "box": [x1, y1, x2, y2]
            })

    # 🔥 FUSIÓN DE CAJAS
    detecciones = fusionar_cajas(detecciones)

    # 🖌️ DIBUJAR CAJAS
    for d in detecciones:
        x1, y1, x2, y2 = d["box"]
        label = f"{d['puntos']} ({int(d['conf']*100)}%)"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    pred_name = f"pred_{os.path.basename(image_path)}"
    pred_path = os.path.join(PRED_FOLDER, pred_name)
    cv2.imwrite(pred_path, img)

    total = sum(d["puntos"] for d in detecciones)

    return {
        "total": total,
        "cantidad": len(detecciones),
        "imagen_pred": pred_path,
        "fichas": [
            {
                "puntos": d["puntos"],
                "confianza": round(d["conf"] * 100, 1),
                "box": d["box"]
            }
            for d in detecciones
        ]
    }


# ================== RUTAS ==================
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

    ext = os.path.splitext(image.filename)[1].lower()
    name = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_FOLDER, name)
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
        resultado.get("imagen_pred", path),
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


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(".", filename)


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model": model is not None
    }


# ================== MAIN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)



