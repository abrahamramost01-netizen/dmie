import os
import uuid
import json
import time
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from openai import OpenAI

# ================= CONFIG =================
WIN_POINTS = 200
MAX_RETRIES = 3
UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no definida")

client = OpenAI(api_key=OPENAI_API_KEY)

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= IA DOMINÓ =================
def calcular_puntos_domino(image_path):
    conn = get_db()
    cur = conn.cursor()

    # ✅ CACHE POR image_path
    cur.execute(
        "SELECT details FROM matches WHERE image_path = %s AND details IS NOT NULL LIMIT 1",
        (image_path,)
    )
    cached = cur.fetchone()
    if cached:
        cur.close()
        conn.close()
        return json.loads(cached[0])

    last_error = None

    for _ in range(MAX_RETRIES):
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()

            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Analiza esta imagen de fichas de dominó DOBLE 9.\n"
                                "Detecta TODAS las fichas visibles, incluso si están juntas.\n"
                                "Para cada ficha indica: valores (lado A y B) y suma.\n"
                                "Devuelve JSON EXACTO con este formato:\n"
                                "{ total: number, fichas: [ { a:number, b:number, suma:number } ] }\n"
                                "NO inventes fichas."
                            )
                        },
                        {
                            "type": "input_image",
                            "image_base64": img_bytes
                        }
                    ]
                }]
            )

            text = response.output_text
            data = json.loads(text)
            cur.close()
            conn.close()
            return data

        except Exception as e:
            last_error = str(e)
            time.sleep(1)

    cur.close()
    conn.close()
    raise RuntimeError(f"IA falló tras {MAX_RETRIES} intentos: {last_error}")

# ================= ROUTES =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
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
    cur.execute("INSERT INTO teams (name) VALUES (%s)", (name,))
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
    image_path = f"{UPLOAD_FOLDER}/{filename}"
    image.save(image_path)

    resultado = calcular_puntos_domino(image_path)
    points = resultado["total"]
    details_json = json.dumps(resultado, ensure_ascii=False)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path, details)
        VALUES (%s, %s, %s, %s)
    """, (team_id, points, image_path, details_json))

    cur.execute("""
        UPDATE teams SET points = points + %s WHERE id = %s
    """, (points, team_id))

    # 🏆 Reset automático
    cur.execute("SELECT id FROM teams WHERE points >= %s", (WIN_POINTS,))
    if cur.fetchone():
        cur.execute("UPDATE teams SET points = 0")

    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= RUN =================
if __name__ == "__main__":
    app.run()



