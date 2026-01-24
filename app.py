import os
import uuid
import json
import hashlib
from datetime import datetime

import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image, ImageDraw
from openai import OpenAI

# ================= CONFIG =================
app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
WIN_POINTS = 200

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= UTIL =================
def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# ================= IA DOMINÓ =================
def calcular_puntos_domino(image_path):
    """
    Devuelve:
    {
      total: int,
      fichas: [
        {x,y,w,h, left, right, puntos}
      ],
      processed_image: "processed/xxx.png"
    }
    """

    file_hash = hash_file(image_path)
    cache_file = f"{PROCESSED_FOLDER}/{file_hash}.json"
    processed_image_path = f"{PROCESSED_FOLDER}/{file_hash}.png"

    # -------- CACHE --------
    if os.path.exists(cache_file) and os.path.exists(processed_image_path):
        with open(cache_file, "r") as f:
            data = json.load(f)
        data["processed_image"] = processed_image_path
        return data

    # -------- OPENAI VISION --------
    with open(image_path, "rb") as img:
        image_bytes = img.read()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Analiza esta imagen de fichas de dominó doble 9.\n"
                        "Detecta TODAS las fichas aunque estén juntas.\n"
                        "Para cada ficha devuelve:\n"
                        "- bounding box (x,y,width,height)\n"
                        "- valor lado izquierdo\n"
                        "- valor lado derecho\n"
                        "- puntos (suma de ambos lados)\n\n"
                        "Devuelve SOLO JSON con este formato:\n"
                        "{ total: number, fichas: [ {x,y,w,h,left,right,puntos} ] }"
                    )
                },
                {
                    "type": "input_image",
                    "image_base64": image_bytes
                }
            ]
        }]
    )

    raw = response.output_text
    data = json.loads(raw)

    # -------- DIBUJAR CAJAS --------
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for f in data["fichas"]:
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
        draw.text((x, y - 15), f'{f["left"]}+{f["right"]}', fill="red")

    img.save(processed_image_path)

    # -------- GUARDAR CACHE --------
    with open(cache_file, "w") as f:
        json.dump(data, f)

    data["processed_image"] = processed_image_path
    return data

# ================= ROUTES =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT m.id, t.name, m.points, m.image_path, m.processed_image
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
    name = request.form["name"].strip()
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
    team_id = int(request.form["team_id"])
    image = request.files.get("image")

    image_path = None
    processed_image = None
    points = 0

    if image and image.filename:
        ext = image.filename.rsplit(".", 1)[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        image_path = f"{UPLOAD_FOLDER}/{filename}"
        image.save(image_path)

        resultado = calcular_puntos_domino(image_path)
        points = resultado["total"]
        processed_image = resultado["processed_image"]

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path, processed_image, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (team_id, points, image_path, processed_image, datetime.utcnow()))

    cur.execute("""
        UPDATE teams
        SET points = points + %s
        WHERE id = %s
    """, (points, team_id))

    # -------- RESET SI LLEGA A 200 --------
    cur.execute("SELECT points FROM teams WHERE id=%s", (team_id,))
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

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)


# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)


