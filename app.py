import os
import uuid
import json
import psycopg2
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
WIN_POINTS = int(os.environ.get("WIN_POINTS", 200))
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= CARGAR MODELO ONNX =================
print(f"🔧 Cargando modelo ONNX desde: {MODEL_PATH}")

try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    model_loaded = True
    print("✅ Modelo ONNX cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando ONNX: {e}")
    net = None
    model_loaded = False


# ================= DB =================
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


# ================= UTILIDADES =================
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = (w1 * h1) + (w2 * h2) - inter

    return inter / union if union > 0 else 0


def fusionar_cajas(detecciones):
    """
    Fusiona detecciones solapadas para evitar contar fichas duplicadas
    """
    detecciones = sorted(detecciones, key=lambda x: x["confianza"], reverse=True)
    resultado = []

    for det in detecciones:
        duplicada = False
        for r in resultado:
            if iou(det["bbox"], r["bbox"]) > IOU_THRESHOLD:
                duplicada = True
                break
        if not duplicada:
            resultado.append(det)

    return resultado


# ================= YOLO ONNX =================
def calcular_puntos_domino(image_path):
    if not model_loaded or net is None:
        return {
            "total": 0,
            "cantidad": 0,
            "fichas": [],
            "error": "Modelo no cargado"
        }

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1 / 255.0,
        size=(640, 640),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward()[0]

    detecciones = []

    for det in outputs:
        scores = det[5:]
        class_id = int(np.argmax(scores))
        confidence = scores[class_id]

        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, bw, bh = det[0:4]
        x = int((cx - bw / 2) * w / 640)
        y = int((cy - bh / 2) * h / 640)
        bw = int(bw * w / 640)
        bh = int(bh * h / 640)

        detecciones.append({
            "clase": class_id,
            "puntos": class_id,  # doble nueve: clase == puntos
            "confianza": round(float(confidence) * 100, 1),
            "bbox": (x, y, bw, bh)
        })

    # 🔥 FUSIÓN DE CAJAS
    detecciones = fusionar_cajas(detecciones)

    total = sum(d["puntos"] for d in detecciones)

    return {
        "total": total,
        "cantidad": len(detecciones),
        "fichas": [
            {
                "clase": d["clase"],
                "puntos": d["puntos"],
                "confianza": d["confianza"]
            } for d in detecciones
        ]
    }


# ================= RUTAS =================
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

    return render_template(
        "index.html",
        teams=teams,
        matches=matches,
        win_points=WIN_POINTS
    )


@app.route("/add_match", methods=["POST"])
def add_match():
    try:
        team_id = int(request.form.get("team_id"))
        image = request.files.get("image")

        if not image:
            return redirect(url_for("index"))

        ext = os.path.splitext(image.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            return redirect(url_for("index"))

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
            path,
            json.dumps(resultado)
        ))

        cur.execute("""
            UPDATE teams SET points = points + %s WHERE id = %s
        """, (resultado["total"], team_id))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("❌ Error add_match:", e)

    return redirect(url_for("index"))


@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }


# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


