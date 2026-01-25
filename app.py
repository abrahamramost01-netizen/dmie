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
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")
WIN_POINTS = int(os.environ.get("WIN_POINTS", 200))
CONF_THRESHOLD = 0.5
IMG_SIZE = 640

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== DATABASE ==================
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================== LOAD ONNX MODEL ==================
print(f"🔧 Cargando modelo ONNX: {MODEL_PATH}")

try:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("✅ Modelo ONNX cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando ONNX: {e}")
    session = None

# ================== YOLO ONNX UTILS ==================
def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def calcular_puntos_domino(image_path):
    """
    Detecta fichas de dominó usando YOLO con fusión de cajas
    para evitar contar fichas duplicadas.
    """

    if model is None:
        return {
            "total": 0,
            "cantidad": 0,
            "fichas": [],
            "error": "Modelo no cargado"
        }

    try:
        results = model(image_path, conf=0.25, iou=0.7, verbose=False)

        detecciones = []

        # ===============================
        # 1️⃣ Extraer detecciones crudas
        # ===============================
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls.item())
                conf = float(box.conf.item())

                detecciones.append({
                    "bbox": (x1, y1, x2, y2),
                    "clase": cls,
                    "conf": conf
                })

        # ===============================
        # 2️⃣ Función IoU
        # ===============================
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH

            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

            union = boxAArea + boxBArea - interArea
            return interArea / union if union > 0 else 0

        # ===============================
        # 3️⃣ Fusión de cajas (anti-duplicados)
        # ===============================
        detecciones.sort(key=lambda x: x["conf"], reverse=True)
        fichas_finales = []

        for det in detecciones:
            duplicada = False
            for f in fichas_finales:
                if iou(det["bbox"], f["bbox"]) > 0.5:
                    duplicada = True
                    break
            if not duplicada:
                fichas_finales.append(det)

        # ===============================
        # 4️⃣ Cálculo de puntos
        # ===============================
        total = 0
        fichas = []

        for f in fichas_finales:
            puntos = f["clase"]  # 👈 AJUSTA si tu dataset usa otro mapeo
            total += puntos

            fichas.append({
                "clase": f["clase"],
                "puntos": puntos,
                "confianza": round(f["conf"] * 100, 1)
            })

        return {
            "total": total,
            "cantidad": len(fichas),
            "fichas": fichas
        }

    except Exception as e:
        return {
            "total": 0,
            "cantidad": 0,
            "fichas": [],
            "error": str(e)
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

    return render_template(
        "index.html",
        teams=teams,
        matches=matches,
        win_points=WIN_POINTS
    )

@app.route("/add_team", methods=["POST"])
def add_team():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("index"))

    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO teams (name) VALUES (%s)", (name,))
    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/delete_team", methods=["POST"])
def delete_team():
    team_id = int(request.form.get("team_id"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM teams WHERE id = %s", (team_id,))
    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for("index"))

@app.route("/add_match", methods=["POST"])
def add_match():
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

    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/health")
def health():
    return {
        "status": "ok",
        "onnx_loaded": session is not None
    }

# ================== START ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

