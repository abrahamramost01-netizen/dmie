import os
import psycopg2
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "super-secret-key"

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id, password FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            return redirect("/")
        return "Usuario o contraseña incorrectos"

    return render_template("login.html")

# ---------- REGISTER ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password) VALUES (%s,%s)",
            (username, password)
        )
        conn.commit()
        conn.close()
        return redirect("/login")

    return render_template("register.html")

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ---------- HOME ----------
@app.route("/")
def index():
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points
        FROM matches m
        JOIN teams t ON t.id = m.team_id
        ORDER BY m.created_at DESC
    """)
    matches = cur.fetchall()

    conn.close()
    return render_template("index.html", teams=teams, matches=matches)

# ---------- CREAR EQUIPO ----------
@app.route("/add_team", methods=["POST"])
def add_team():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO teams (name) VALUES (%s)", (request.form["name"],))
    conn.commit()
    conn.close()
    return redirect("/")

# ---------- EDITAR PUNTOS ----------
@app.route("/edit_team_points", methods=["POST"])
def edit_team_points():
    team_id = request.form["team_id"]
    points = int(request.form["points"])

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT points, wins FROM teams WHERE id=%s", (team_id,))
    current_points, wins = cur.fetchone()

    new_points = current_points + points

    if new_points >= 200:
        wins += 1
        new_points = 0

    cur.execute(
        "UPDATE teams SET points=%s, wins=%s WHERE id=%s",
        (new_points, wins, team_id)
    )

    conn.commit()
    conn.close()
    return redirect("/")

# ---------- ELIMINAR EQUIPO ----------
@app.route("/delete_team", methods=["POST"])
def delete_team():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM teams WHERE id=%s", (request.form["team_id"],))
    conn.commit()
    conn.close()
    return redirect("/")

# ---------- AGREGAR PARTIDA ----------
@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = request.form["team_id"]
    points = int(request.form["points"])

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO matches (team_id, points) VALUES (%s,%s)",
        (team_id, points)
    )

    cur.execute(
        "UPDATE teams SET points = points + %s WHERE id=%s",
        (points, team_id)
    )

    conn.commit()
    conn.close()
    return redirect("/")

if __name__ == "__main__":
    app.run()



