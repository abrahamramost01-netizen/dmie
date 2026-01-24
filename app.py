import os
import uuid
import base64
import json
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from openai import OpenAI

# ================= CONFIG =================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
WIN_POINTS = 200

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
def calcular_puntos_domino(image_path: str) -> tuple[int, str]:
    """
    Devuelve:
    - total_puntos (int)
    - detalle (str)
    """

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
Eres un sistema experto en dominó DOBLE 9.

Reglas OBLIGATORIAS:
- Cada ficha tiene exactamente DOS valores.
- Los valores posibles son del 0 al 9.
- Las fichas pueden estar juntas o tocándose.
- Detecta TODAS las fichas visibles.
- Suma TODOS los puntos visibles.

Devuelve SOLO JSON válido con esta estructura EXACTA:

{
  "total": 45,
  "fichas": [
    {"lado_1": 6, "lado_2": 9, "suma": 15},
    {"lado_1": 3, "lado_2": 5, "suma": 8}
  ],
  "explicacion": "Texto breve explicando el cálculo"
}

NO agregues texto fuera del JSON.
"""

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
        max_output_tokens=500
    )

    raw = response.output_text.strip()

    try:
        data = json.loads(raw)
        total = int(data.get("total", 0))
        detalle = data.get("explicacion", "")
        return total, detalle
    except Exception:
        # Fallback seguro
        return 0, "No se pudo interpretar la imagen correctamente"

# ================= RUTAS =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points, m.created_at, m.image_path, m.detail
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

    if not image or not image.filename:
        return redirect(url_for("index"))

    ext = image.filename.rsplit(".", 1)[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    image_path = f"{UPLOAD_FOLDER}/{filename}"
    image.save(image_path)

    points, detail = calcular_puntos_domino(image_path)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path, detail)
        VALUES (%s, %s, %s, %s)
    """, (team_id, points, image_path, detail))

    cur.execute("""
        UPDATE teams
        SET points = points + %s
        WHERE id = %s
    """, (points, team_id))

    # 🔁 RESET AUTOMÁTICO AL LLEGAR A 200
    cur.execute("""
        SELECT id FROM teams WHERE points >= %s
    """, (WIN_POINTS,))
    winners = cur.fetchall()

    if winners:
        cur.execute("UPDATE teams SET points = 0")

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)

