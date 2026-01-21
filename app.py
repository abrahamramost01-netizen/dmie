import os
import psycopg2
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    db = get_db()
    cur = db.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS teams (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        points INTEGER DEFAULT 0
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        id SERIAL PRIMARY KEY,
        team_id INTEGER,
        points INTEGER
    );
    """)

    db.commit()
    cur.close()
    db.close()

@app.route("/")
def index():
    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT * FROM teams")
    teams = cur.fetchall()

    cur.execute("SELECT * FROM matches ORDER BY id DESC")
    matches = cur.fetchall()

    cur.close()
    db.close()

    return render_template("index.html", teams=teams, matches=matches)

@app.route("/add_team", methods=["POST"])
def add_team():
    name = request.form["name"]

    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO teams (name) VALUES (%s)", (name,))
    db.commit()
    cur.close()
    db.close()

    return redirect("/")

@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = request.form["team_id"]
    points = int(request.form["points"])

    db = get_db()
    cur = db.cursor()

    cur.execute(
        "INSERT INTO matches (team_id, points) VALUES (%s, %s)",
        (team_id, points)
    )
    cur.execute(
        "UPDATE teams SET points = points + %s WHERE id = %s",
        (points, team_id)
    )

    db.commit()
    cur.close()
    db.close()

    return redirect("/")

if __name__ == "__main__":
    init_db()
    app.run()
