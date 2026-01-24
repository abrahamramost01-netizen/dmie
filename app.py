import os
import psycopg2
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ===================== HOME =====================
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

# ===================== CREAR EQUIPO =====================
@app.route("/add_team", methods=["POST"])
def add_team():
    name = request.form["name"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO teams (name, points, wins) VALUES (%s, 0, 0)",
        (name,)
    )
    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")

# ===================== EDITAR PUNTOS =====================
@app.route("/edit_team_points", methods=["POST"])
def edit_team_points():
    team_id = request.form["team_id"]
    points = int(request.form["points"])

    conn = get_db()
    cur = conn.cursor()

    # Si llega a 200 o más → gana partida y se reinicia
    if points >= 200:
        cur.execute("""
            UPDATE teams
            SET points = 0,
                wins = wins + 1
            WHERE id = %s
        """, (team_id,))
    else:
        cur.execute("""
            UPDATE teams
            SET points = %s
            WHERE id = %s
        """, (points, team_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")

# ===================== ELIMINAR EQUIPO =====================
@app.route("/delete_team", methods=["POST"])
def delete_team():
    team_id = request.form["team_id"]

    conn = get_db()
    cur = conn.cursor()

    cur.execute("DELETE FROM matches WHERE team_id = %s", (team_id,))
    cur.execute("DELETE FROM teams WHERE id = %s", (team_id,))

    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")

# ===================== AGREGAR PARTIDA =====================
@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = request.form["team_id"]
    points = int(request.form["points"])

    conn = get_db()
    cur = conn.cursor()

    # Guardar historial
    cur.execute(
        "INSERT INTO matches (team_id, points) VALUES (%s, %s)",
        (team_id, points)
    )

    # Actualizar puntos del equipo
    cur.execute(
        "UPDATE teams SET points = points + %s WHERE id = %s",
        (points, team_id)
    )

    # Verificar si llegó a 200
    cur.execute("SELECT points FROM teams WHERE id = %s", (team_id,))
    total = cur.fetchone()[0]

    if total >= 200:
        cur.execute("""
            UPDATE teams
            SET points = 0,
                wins = wins + 1
            WHERE id = %s
        """, (team_id,))

    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")


