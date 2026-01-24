import os
import uuid
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from openai import OpenAI

# ================= CONFIG =================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
WIN_POINTS = 200

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definida")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= IA – CONTAR PUNTOS DOMINÓ =================
def calcular_puntos_domino(image_path: str) -> int:
    """
    Usa IA para contar puntos de dominó doble-9.
    Considera fichas juntas y suma TODOS los lados visibles.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Esta imagen contiene fichas de dominó DOBLE-9. "
                        "Las fichas pueden estar juntas o tocándose. "
                        "Detecta TODAS las fichas visibles. "
                        "Cada ficha tiene dos lados con puntos (0 a 9). "
                        "Suma TODOS los puntos de TODAS las fichas. "
                        "Devuelve SOLO un número entero. Nada más."
                    )
                },
                {
                    "type": "input_image",
                    "image_base64": image_bytes
                }
            ]
        }]
    )

    texto = response.output_text.strip()

    # Seguridad: solo números
    try:
        return int("".join(c for c in texto if c.isdigit()))
    except:
        return 0

# ================= ROUTES =================
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

    return render_template(
        "index.html",
        teams=teams,
        matches=matches,
        win_points=WIN_POINTS
    )

@app.route("/add_team", methods=["POST"])
def add_team():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("index"))

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO teams (name, points, wins) VALUES (%s, 0, 0)",
        (name,)
    )
    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for("index"))

@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = int(request.form.get("team_id"))

    image = request.files.get("image")
    points = 0
    image_path = None

    # ================= FOTO + IA =================
    if image and image.filename:
        ext = image.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        image_path = f"{UPLOAD_FOLDER}/{filename}"
        image.save(image_path)

        points = calcular_puntos_domino(image_path)

    conn = get_db()
    cur = conn.cursor()

    # Guardar partida
    cur.execute(
        "INSERT INTO matches (team_id, points, image_path) VALUES (%s, %s, %s)",
        (team_id, points, image_path)
    )

    # Sumar puntos
    cur.execute(
        "UPDATE teams SET points = points + %s WHERE id = %s",
        (points, team_id)
    )

    # Verificar victoria
    cur.execute("SELECT points FROM teams WHERE id = %s", (team_id,))
    total = cur.fetchone()[0]

    if total >= WIN_POINTS:
        # sumar victoria y resetear
        cur.execute(
            "UPDATE teams SET wins = wins + 1, points = 0"
        )

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)


