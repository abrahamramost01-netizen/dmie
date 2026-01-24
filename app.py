import os
import uuid
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
UPLOAD_FOLDER = "uploads"

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definida")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


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

    return render_template("index.html", teams=teams, matches=matches)


@app.route("/add_team", methods=["POST"])
def add_team():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("index"))

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO teams (name) VALUES (%s)",
        (name,)
    )

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))


@app.route("/add_match", methods=["POST"])
def add_match():
    try:
        team_id = int(request.form.get("team_id"))
        points = int(request.form.get("points"))
    except (TypeError, ValueError):
        return redirect(url_for("index"))

    image = request.files.get("image")
    image_path = None

    if image and image.filename:
        ext = image.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        image_path = f"{UPLOAD_FOLDER}/{filename}"
        image.save(image_path)

    conn = get_db()
    cur = conn.cursor()

    # Guardar partida
    cur.execute("""
        INSERT INTO matches (team_id, points, image_path)
        VALUES (%s, %s, %s)
    """, (team_id, points, image_path))

    # Sumar puntos y obtener nuevo total
    cur.execute("""
        UPDATE teams
        SET points = points + %s
        WHERE id = %s
        RETURNING points
    """, (points, team_id))

    new_points = cur.fetchone()[0]

    # 🔥 REGLA DE 200 PUNTOS
    if new_points >= 200:
        # Sumar partida ganada
        cur.execute("""
            UPDATE teams
            SET wins = wins + 1
            WHERE id = %s
        """, (team_id,))

        # Resetear puntos de TODOS los equipos
        cur.execute("""
            UPDATE teams
            SET points = 0
        """)

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))


@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run()


