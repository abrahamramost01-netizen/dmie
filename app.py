import os
import uuid
import json
import base64
import hashlib
import time
from datetime import datetime

import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from PIL import Image, ImageDraw
from openai import OpenAI

# ================= CONFIG =================
WIN_POINTS = 200
MAX_RETRIES = 3

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
JSON_FOLDER = "matches_json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definida")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está definida")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= IA DOMINÓ =================
def calcular_puntos_domino(image_path):
    """
    Devuelve:
    {
      total: int,
      fichas: [{a,b,suma,box}],
      processed_image: str
    }
    """

    # ---- Cache por hash ----
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    img_hash = hashlib.sha256(img_bytes).hexdigest()
    cache_json = os.path.join(JSON_FOLDER, f"{img_hash}.json")

    if os.path.exists(cache_json):
        with open(cache_json, "r") as f:
            return json.load(f)

    # ---- Base64 ----
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = """
Eres un sistema experto en dominó DOBLE-9.

Analiza la imagen y detecta TODAS las fichas visibles.
Reglas IMPORTANTES:
- Las fichas pueden estar juntas o tocándose
- Cada ficha tiene DOS valores (0 a 9)
- Suma ambos lados
- No inventes fichas
- Si hay duda, NO cuentes esa ficha
- Devuelve cajas ajustadas a cada ficha

Devuelve SOLO JSON con este formato:

{
  "fichas": [
    {
      "a": 6,
      "b": 4,
      "suma": 10,
      "box": [x1, y1, x2, y2]
    }
  ],
  "total": 42
}
"""

    last_error = None

    for intento in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    ]
                }],
                response_format={"type": "json_object"}
            )

            data = json.loads(response.output_text)

            # ---- Dibujar cajas ----
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            for f in data["fichas"]:
                draw.rectangle(f["box"], outline="red", width=4)
                draw.text((f["box"][0], f["box"][1] - 15),
                          f'{f["a"]}+{f["b"]}={f["suma"]}',
                          fill="red")

            processed_name = f"{uuid.uuid4()}.jpg"
            processed_path = os.path.join(PROCESSED_FOLDER, processed_name)
            img.save(processed_path)

            data["processed_image"] = processed_name
            data["timestamp"] = datetime.utcnow().isoformat()

            # ---- Guardar cache ----
            with open(cache_json, "w") as f:
                json.dump(data, f, indent=2)

            return data

        except Exception as e:
            last_error = e
            time.sleep(1)

    raise RuntimeError(f"IA falló tras reintentos: {last_error}")

# ================= ROUTES =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points, m.image_path, m.details_json, m.processed_image
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

    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    result = calcular_puntos_domino(image_path)

    points = result["total"]

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path, details_json, processed_image)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        team_id,
        points,
        image_path,
        json.dumps(result),
        result["processed_image"]
    ))

    cur.execute("""
        UPDATE teams SET points = points + %s WHERE id = %s
    """, (points, team_id))

    # ---- Reset si llega a 200 ----
    cur.execute("SELECT points FROM teams WHERE id = %s", (team_id,))
    total = cur.fetchone()[0]

    if total >= WIN_POINTS:
        cur.execute("UPDATE teams SET points = 0")

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(".", filename)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)




