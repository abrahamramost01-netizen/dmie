import os
import psycopg2
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Railway proporciona esta variable automáticamente
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(
        DATABASE_URL,
        sslmode="require"
    )

@app.route("/")
def index():
    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points, m.created_at
        FROM matches m
        JOIN teams t ON m.team_id = t.id
        ORDER BY m.created_at DESC
    """)
    matches = cur.fetchall()

    cur.close()
    db.close()

    return render_template(
        "index.html",
        teams=teams,
        matches=matches
    )

@app.route("/add_team", methods=["POST"])
def add_team():
    name = request.form["name"].strip()

    if name == "":
        return redirect(url_for("index"))

    db = get_db()
    cur = db.cursor()

    cur.execute(
        "INSERT INTO teams (name) VALUES (%s)",
        (name,)
    )

    db.commit()
    cur.close()
    db.close()

    return redirect(url_for("index"))

@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = request.form["team_id"]
    points = request.form["points"]

    try:
        points = int(points)
    except ValueError:
        return redirect(url_for("index"))

    if points <= 0:
        return redirect(url_for("index"))

    db = get_db()
    cur = db.cursor()

    # Guardar la partida
    cur.execute(
        "INSERT INTO matches (team_id, points) VALUES (%s, %s)",
        (team_id, points)
    )

    # Sumar puntos al equipo
    cur.execute(
        "UPDATE teams SET points = points + %s WHERE id = %s",
        (points, team_id)
    )

    db.commit()
    cur.close()
    db.close()

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run()


