import os
import psycopg2
from flask import Flask, render_template, request, redirect, url_for

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

@app.route("/add_team", methods=["POST"])
def add_team():
    name = request.form["name"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO teams (name, points) VALUES (%s, 0)",
        (name,)
    )
    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/edit_team_points", methods=["POST"])
def edit_team_points():
    team_id = request.form["team_id"]
    points = request.form["points"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE teams SET points = %s WHERE id = %s",
        (points, team_id)
    )
    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

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

    return redirect(url_for("index"))

@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = request.form["team_id"]
    points = int(request.form["points"])

    conn = get_db()
    cur = conn.cursor()

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

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

