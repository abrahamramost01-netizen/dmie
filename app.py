import os
import psycopg2
from flask import Flask, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "secret")

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ---------------- AUTH ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id, password FROM users WHERE username=%s",
                    (request.form["username"],))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[1], request.form["password"]):
            session["user_id"] = user[0]
            return redirect("/")
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password) VALUES (%s, %s)",
        (request.form["username"],
         generate_password_hash(request.form["password"]))
    )
    conn.commit()
    conn.close()
    return redirect("/login")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ---------------- MAIN ----------------
@app.route("/")
def index():
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT id, name, points, wins FROM teams WHERE user_id=%s",
        (session["user_id"],)
    )
    teams = cur.fetchall()

    cur.execute("""
        SELECT t.name, m.points
        FROM matches m
        JOIN teams t ON t.id = m.team_id
        WHERE t.user_id=%s
        ORDER BY m.created_at DESC
    """, (session["user_id"],))
    matches = cur.fetchall()

    conn.close()
    return render_template("index.html", teams=teams, matches=matches)

# ---------------- TEAMS ----------------
@app.route("/add_team", methods=["POST"])
def add_team():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO teams (user_id, name) VALUES (%s, %s)",
        (session["user_id"], request.form["name"])
    )
    conn.commit()
    conn.close()
    return redirect("/")

@app.route("/edit_team_points", methods=["POST"])
def edit_points():
    points = int(request.form["points"])
    team_id = request.form["team_id"]

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT points, wins FROM teams WHERE id=%s", (team_id,))
    current, wins = cur.fetchone()
    new_points = current + points

    if new_points >= 200:
        wins += 1
        new_points = 0

    cur.execute(
        "UPDATE teams SET points=%s, wins=%s WHERE id=%s",
        (new_points, wins, team_id)
    )

    cur.execute(
        "INSERT INTO matches (team_id, points) VALUES (%s, %s)",
        (team_id, points)
    )

    conn.commit()
    conn.close()
    return redirect("/")

@app.route("/delete_team", methods=["POST"])
def delete_team():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM teams WHERE id=%s", (request.form["team_id"],))
    conn.commit()
    conn.close()
    return redirect("/")

