import os
import uuid
import json
import base64
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from openai import OpenAI

# ================= CONFIG =================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
CACHE_FOLDER = "cache"
WIN_POINTS = 200

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no definida")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no definida")

client = OpenAI(api_key=OPENAI_API_KEY)

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= IA DOMINÓ =================
def calcular_puntos_domino(image_path, retries=2):
    cache_file = os.path.join(CACHE_FOLDER, os.path.basename(image_path) + ".json")

    # 🔹 CACHE
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    with open(image_path, "rb") as img:
        image_b64 = base64.b64encode(img.read()).decode("utf-8")

    prompt = """
Analiza esta imagen con fichas de dominó DOBLE-9.

REGLAS:
- Cada ficha tiene dos lados con valores de 0 a 9
- Las fichas pueden estar juntas o tocándose
- Detecta TODAS las fichas visibles
- Para cada ficha devuelve:
  - valores: [lado1, lado2]
  - suma: lado1 + lado2
- Calcula la suma TOTAL

RESPONDE SOLO EN JSON con este formato exacto:
{
  "total": number,
  "fichas": [
    {"lados": [n, n], "suma": n}
  ]
}
"""

    for intento in range(retries + 1):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_base64": image_b64
                            }
                        ]
                    }
                ],
                max_output_tokens=500,
            )

            text = response.output_text.strip()
            data = json.loads(text)

            # Guardar cache
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            return data

        except Exception as e:
            if intento == retries:
                raise RuntimeError(f"IA falló: {e}")

    return {"total": 0, "fichas": []}

# ================= RUTAS =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points, m.created_at, m.image_path, m.detail_json
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
    cur.execute("INSERT INTO teams (name, points) VALUES (%s, 0)", (name,))
    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/add_match", methods=["POST"])
def add_match():
    team_id = int(request.form.get("team_id"))
    image = request.files.get("image")

    image_path = None
    detail = {"total": 0, "fichas": []}
    points = 0

    if image and image.filename:
        ext = image.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)

        detail = calcular_puntos_domino(image_path)
        points = detail["total"]

    conn = get_db()
    cur = conn.cursor()

    # Insert match
    cur.execute("""
        INSERT INTO matches (team_id, points, image_path, detail_json)
        VALUES (%s, %s, %s, %s)
    """, (team_id, points, image_path, json.dumps(detail)))

    # Update team points
    cur.execute("""
        UPDATE teams
        SET points = points + %s
        WHERE id = %s
        RETURNING points
    """, (points, team_id))

    new_points = cur.fetchone()[0]

    # 🏆 WIN CONDITION
    if new_points >= WIN_POINTS:
        cur.execute("UPDATE teams SET points = 0")

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)




