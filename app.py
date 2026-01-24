import os
import uuid
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
import pytesseract
import re

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
UPLOAD_FOLDER = "uploads"
WIN_POINTS = 200

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definida")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


# ========================= OCR DOMINÓ =========================
def ocr_domino_points(image_path):
    """
    OCR simple para leer números de fichas de dominó.
    Devuelve puntos detectados o None.
    """
    try:
        img = Image.open(image_path).convert("L")
        text = pytesseract.image_to_string(img, config="--psm 6 digits")
        nums = re.findall(r"\d+", text)
        if nums:
            return sum(int(n) for n in nums)
    except Exception:
        pass
    return None


# ========================= HOME =========================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points, wins FROM teams ORDER BY id")
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

    return render_template("index.html", teams=teams, matches=matches, win_points=WIN_POINTS)


# ========================= EQUIPOS =========================
@app.route("/add_team", methods=["POST"])
def add_team():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("index"))

    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO teams (name, points, wins) VALUES (%s, 0, 0)", (name,))
    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for("index"))


# ========================= PARTIDA =========================
@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = int(request.form.get("team_id"))
    points = request.form.get("points")

    image = request.files.get("image")
    image_path = None

    if image and image.filename:
        ext = image.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        image_path = f"{UPLOAD_FOLDER}/{filename}"
        image.save(image_path)

        # OCR automático si no se escribió puntaje
        if not points:
            detected = ocr_domino_points(image_path)
            if detected:
                points = detected

    try:
        points = int(points)
    except:
        return redirect(url_for("index"))

    conn = get_db()
    cur = conn.cursor()

    # Guardar partida
    cur.execute("""
        INSERT INTO matches (team_id, points, image_path)
        VALUES (%s, %s, %s)
    """, (team_id, points, image_path))

    # Actualizar puntos
    cur.execute("""
        UPDATE teams
        SET points = points + %s
        WHERE id = %s
        RETURNING points
    """, (points, team_id))

    new_points = cur.fetchone()[0]

    # ¿Ganó?
    if new_points >= WIN_POINTS:
        cur.execute("UPDATE teams SET wins = wins + 1 WHERE id = %s", (team_id,))
        cur.execute("UPDATE teams SET points = 0")  # reset ambos

    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for("index"))


# ========================= UPLOADS =========================
@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run()

