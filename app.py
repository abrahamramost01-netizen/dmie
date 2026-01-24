import os
import psycopg2
from flask import Flask, render_template, request, redirect, abort

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
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

# -------- EQUIPOS --------

@app.route("/add_team", methods=["POST"])
def add_team():
    name = request.form.get("name", "").strip()

    if not name:
        abort(400)

    conn = get_db()
    cur = conn.cursor()

    # evitar nombres duplicados
    cur.execute("SELECT 1 FROM teams WHERE LOWER(name)=LOWER(%s)", (name,))
    if cur.fetchone():
        cur.close()
        conn.close()
        return redirect("/")

    cur.execute(
        "INSERT INTO teams (name, points) VALUES (%s, 0)",
        (name,)
    )
    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")

@app.route("/edit_team_points", methods=["POST"])
def edit_team_points():
    try:
        team_id = int(request.form["team_id"])
        points = int(request.form["points"])
    except:
        abort(400)

    if points < 0:
        return redirect("/")

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "UPDATE teams SET points=%s WHERE id=%s",
        (points, team_id)
    )

    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")

@app.route("/delete_team", methods=["POST"])
def delete_team():
    try:
        team_id = int(request.form["team_id"])
    except:
        abort(400)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("DELETE FROM matches WHERE team_id=%s", (team_id,))
    cur.execute("DELETE FROM teams WHERE id=%s", (team_id,))

    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")

# -------- PARTIDAS --------

@app.route("/add_match", methods=["POST"])
def add_match():
    try:
        team_id = int(request.form["team_id"])
        points = int(request.form["points"])
    except:
        abort(400)

    if points <= 0:
        return redirect("/")

    conn = get_db()
    cur = conn.cursor()

    # verificar equipo existe
    cur.execute("SELECT 1 FROM teams WHERE id=%s", (team_id,))
    if not cur.fetchone():
        cur.close()
        conn.close()
        abort(400)

    cur.execute(
        "INSERT INTO matches (team_id, points) VALUES (%s, %s)",
        (team_id, points)
    )

    cur.execute(
        "UPDATE teams SET points = points + %s WHERE id = %s",
        (points, team_id)
    )

    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)

