import os
import uuid
import json
import base64
import hashlib
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from openai import OpenAI

# ================= CONFIG =================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
WIN_POINTS = 200
MAX_RETRIES = 3

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
def calcular_puntos_domino(image_path):
    """
    Devuelve:
    {
      total: int,
      fichas: [{a,b,suma,x,y,w,h}],
      intentos: int
    }
    """

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_hash = hashlib.sha256(image_bytes).hexdigest()

    conn = get_db()
    cur = conn.cursor()

    # 🔹 CACHE
    cur.execute(
        "SELECT details FROM matches WHERE image_hash = %s LIMIT 1",
        (image_hash,)
    )
    row = cur.fetchone()
    if row:
        cur.close()
        conn.close()
        return json.loads(row[0])

    image_b64 = base64.b64encode(image_bytes).decode()
    image_data_url = f"data:image/jpeg;base64,{image_b64}"

    last_error = None

    for intento in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Analiza una imagen de fichas de dominó DOBLE 9.\n"
                                "Reglas IMPORTANTES:\n"
                                "- Cada ficha tiene dos valores entre 0 y 9\n"
                                "- Puede haber fichas juntas o tocándose\n"
                                "- No inventes fichas\n"
                                "- Devuelve JSON ESTRICTO\n\n"
                                "Formato:\n"
                                "{\n"
                                "  \"total\": number,\n"
                                "  \"fichas\": [\n"
                                "    {\"a\":int,\"b\":int,\"suma\":int,\"x\":int,\"y\":int,\"w\":int,\"h\":int}\n"
                                "  ]\n"
                                "}"
                            )
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url
                        }
                    ]
                }]
            )

            text = response.output_text
            data = json.loads(text)
            data["intentos"] = intento

            cur.close()
            conn.close()
            return data

        except Exception as e:
            last_error = str(e)

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

    if not image or not image.filename:
        return redirect(url_for("index"))

    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    resultado = calcular_puntos_domino(image_path)
    total = resultado["total"]

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path, details, image_hash)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        team_id,
        total,
        image_path,
        json.dumps(resultado),
        hashlib.sha256(open(image_path, "rb").read()).hexdigest()
    ))

    cur.execute("""
        UPDATE teams SET points = points + %s WHERE id = %s
    """, (total, team_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)







