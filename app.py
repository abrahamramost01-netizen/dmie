import os
import uuid
import psycopg2
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
UPLOAD_FOLDER = "uploads"

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definida")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= VISION (DOMINÓ) =================
def contar_puntos_domino(ruta_imagen):
    img = cv2.imread(ruta_imagen)

    if img is None:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=4,
        maxRadius=15
    )

    if circles is None:
        return 0

    return len(circles[0])

# ================= ROUTES =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points, wins FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points, m.created_at
        FROM matches m
        JOIN teams t ON t.id = m.team_id
        ORDER BY m.created_at DESC
    """)
    matches = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("index.html", teams=teams, matches=matches)

# ================= TEAMS =================
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

# ================= MATCH (MANUAL O FOTO) =================
@app.route("/add_match", methods=["POST"])
def add_match():
    try:
        team_id = int(request.form.get("team_id"))
    except (TypeError, ValueError):
        return redirect(url_for("index"))

    points_input = request.form.get("points")
    image = request.files.get("image")

    image_path = None

    # 1️⃣ PRIORIDAD: puntos escritos
    if points_input:
        try:
            points = int(points_input)
        except ValueError:
            return redirect(url_for("index"))

    # 2️⃣ SI NO, CONTAR DESDE FOTO
    elif image and image.filename:
        ext = image.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)

        points = contar_puntos_domino(image_path)

        if points <= 0:
            return redirect(url_for("index"))

    else:
        return redirect(url_for("index"))

    conn = get_db()
    cur = conn.cursor()

    # Guardar partida
    cur.execute(
        "INSERT INTO matches (team_id, points) VALUES (%s, %s)",
        (team_id, points)
    )

    # Sumar puntos
    cur.execute(
        "UPDATE teams SET points = points + %s WHERE id = %s",
        (points, team_id)
    )

    # 🔥 GANADOR A 200
    cur.execute("SELECT points FROM teams WHERE id = %s", (team_id,))
    total = cur.fetchone()[0]

    if total >= 200:
        cur.execute("""
            UPDATE teams
            SET wins = wins + 1,
                points = 0
            WHERE id = %s
        """, (team_id,))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

# ================= UPLOADS =================
@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= RUN =================
if __name__ == "__main__":
    app.run()



