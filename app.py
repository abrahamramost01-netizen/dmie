import os
import uuid
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
import cv2

UPLOAD_FOLDER = "uploads"
WIN_POINTS = 200

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ====== DB ======
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ====== YOLO ======
model = YOLO("best.pt")  # <-- TU MODELO ENTRENADO

def calcular_puntos_domino(image_path):
    results = model(image_path, conf=0.5)
    total = 0

    for r in results:
        for cls in r.boxes.cls:
            total += int(cls.item())  # clase = puntos

    return total

# ====== RUTAS ======
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

    if not image:
        return redirect(url_for("index"))

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    points = calcular_puntos_domino(path)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path)
        VALUES (%s, %s, %s)
    """, (team_id, points, path))

    cur.execute("""
        UPDATE teams SET points = points + %s WHERE id = %s
    """, (points, team_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
