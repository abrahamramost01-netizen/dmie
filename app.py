import os
import uuid
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO

# ================= CONFIG =================
WIN_POINTS = 200
UPLOAD_FOLDER = "uploads"
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definida")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= YOLO =================
# Carga del modelo entrenado
model = YOLO("best.pt")

def calcular_puntos_domino(image_path: str) -> int:
    """
    Detecta fichas de dominó con YOLO y suma los puntos.
    Cada clase representa un valor (0–6).
    """
    results = model(image_path, conf=0.4, verbose=False)
    total = 0

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            clase = int(box.cls[0])
            total += clase

    return total

# ================= RUTAS =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points, m.created_at, m.image_path
        FROM matches m
        JOIN teams t ON t.id = m.team_id
        ORDER BY m.created_at DESC
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

@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = int(request.form.get("team_id"))
    image = request.files.get("image")

    if not image or not image.filename:
        return redirect(url_for("index"))

    ext = image.filename.rsplit(".", 1)[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    # 🔥 YOLO calcula los puntos
    points = calcular_puntos_domino(image_path)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path)
        VALUES (%s, %s, %s)
    """, (team_id, points, image_path))

    cur.execute("""
        UPDATE teams
        SET points = points + %s
        WHERE id = %s
    """, (points, team_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
