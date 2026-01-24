import os
import json
import uuid
import base64
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from openai import OpenAI
from PIL import Image

# ================= CONFIG =================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
WIN_POINTS = 200
MAX_RETRIES = 3

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definida")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está definida")

client = OpenAI(api_key=OPENAI_API_KEY)

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= IA DOMINÓ =================
def calcular_puntos_domino(image_path):
    """
    Usa IA para detectar fichas de dominó doble 9,
    devuelve puntos totales + detalle ficha por ficha.
    Cachea resultado en JSON.
    """

    cache_file = image_path + ".json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = """
Eres un sistema experto en dominó DOBLE 9.

Analiza la imagen:
- Puede haber muchas fichas juntas o tocándose
- Cada ficha tiene dos lados (0 a 9)
- Cuenta TODAS las fichas visibles
- Suma todos los puntos correctamente

Devuelve SOLO JSON con este formato EXACTO:

{
  "total": number,
  "fichas": [
    { "lado_a": number, "lado_b": number, "suma": number }
  ]
}

No escribas texto adicional.
"""

    last_error = None

    for intento in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_base64": image_base64
                        }
                    ]
                }],
                max_output_tokens=300
            )

            text = response.output_text
            data = json.loads(text)

            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            return data

        except Exception as e:
            last_error = e

    raise RuntimeError(f"IA falló tras {MAX_RETRIES} intentos: {last_error}")

# ================= ROUTES =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points, wins FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points, m.image_path, m.details
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
    cur.execute("INSERT INTO teams (name, points, wins) VALUES (%s, 0, 0)", (name,))
    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for("index"))

@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = int(request.form.get("team_id"))
    image = request.files.get("image")

    if not image:
        return redirect(url_for("index"))

    ext = image.filename.rsplit(".", 1)[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    resultado = calcular_puntos_domino(image_path)
    puntos = int(resultado["total"])

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path, details)
        VALUES (%s, %s, %s, %s)
    """, (
        team_id,
        puntos,
        image_path,
        json.dumps(resultado)
    ))

    cur.execute("""
        UPDATE teams
        SET points = points + %s
        WHERE id = %s
    """, (puntos, team_id))

    # verificar ganador
    cur.execute("SELECT points FROM teams WHERE id = %s", (team_id,))
    total_points = cur.fetchone()[0]

    if total_points >= WIN_POINTS:
        cur.execute("""
            UPDATE teams
            SET wins = wins + 1,
                points = 0
        """)
    
    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(".", filename)

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)






