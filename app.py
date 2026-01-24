import os
import psycopg2
from flask import Flask, render_template, request, redirect, session, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


# ================= INDEX =================
@app.route("/")
def index():
    user = session.get("user")
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points, wins FROM teams ORDER BY id")
    teams = [
        {"id": r[0], "name": r[1], "points": r[2], "wins": r[3]}
        for r in cur.fetchall()
    ]

    cur.execute("""
        SELECT m.id, t.name, m.points
        FROM matches m
        JOIN teams t ON t.id = m.team_id
        ORDER BY m.id DESC
    """)
    matches = [{"id": r[0], "team_name": r[1], "points": r[2]} for r in cur.fetchall()]

    cur.close()
    conn.close()

    return render_template("index.html", teams=teams, matches=matches, user=user)


# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return redirect("/")

    username = request.form["username"]
    password = request.form["password"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM users WHERE username=%s AND password=%s",
        (username, password),
    )
    user = cur.fetchone()
    cur.close()
    conn.close()

    if user:
        session["user"] = username

    return redirect("/")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ================= TEAMS =================
@app.route("/add_team", methods=["POST"])
def add_team():
    if "user" not in session:
        return redirect("/")

    name = request.form["name"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO teams (name, points, wins) VALUES (%s, 0, 0)",
        (name,),
    )
    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")


@app.route("/edit_team_points", methods=["POST"])
def edit_team_points():
    if "user" not in session:
        return redirect("/")

    team_id = request.form["team_id"]
    points = int(request.form["points"])

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "UPDATE teams SET points=%s WHERE id=%s",
        (points, team_id),
    )

    # regla 200 puntos
    if points >= 200:
        cur.execute(
            "UPDATE teams SET wins = wins + 1, points = 0 WHERE id=%s",
            (team_id,),
        )

    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")


@app.route("/delete_team", methods=["POST"])
def delete_team():
    if "user" not in session:
        return redirect("/")

    team_id = request.form["team_id"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM matches WHERE team_id=%s", (team_id,))
    cur.execute("DELETE FROM teams WHERE id=%s", (team_id,))
    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")


# ================= MATCHES =================
@app.route("/add_match", methods=["POST"])
def add_match():
    if "user" not in session:
        return redirect("/")

    team_id = request.form["team_id"]
    points = int(request.form["points"])

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO matches (team_id, points) VALUES (%s, %s)",
        (team_id, points),
    )

    cur.execute(
        "UPDATE teams SET points = points + %s WHERE id=%s",
        (points, team_id),
    )

    # comprobar 200 puntos
    cur.execute("SELECT points FROM teams WHERE id=%s", (team_id,))
    total = cur.fetchone()[0]

    if total >= 200:
        cur.execute(
            "UPDATE teams SET wins = wins + 1, points = 0 WHERE id=%s",
            (team_id,),
        )

    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)

