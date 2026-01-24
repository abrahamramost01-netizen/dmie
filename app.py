import os
import uuid
import base64
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from openai import OpenAI

WIN_POINTS = 200
app = Flask(__name__)

# ================= CONFIG =================
DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
UPLOAD_FOLDER = "uploads"

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definida")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está definida")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
client = OpenAI(api_key=OPENAI_API_KEY)

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= IA DOMINÓ =================
def calcular_puntos_domino(image_path: str) -> int:
    """
    Envía la imagen a OpenAI Vision y devuelve los puntos detectados.
    """
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un árbitro experto en dominó. "
                    "Recibirás una foto con fichas de dominó visibles. "
                    "Debes calcular la suma total de puntos. "
                    "Devuelve SOLO un número entero. "
                    "No expliques nada."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Calcula los puntos de esta jugada de dominó"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=10
    )

    try:
        return int(response.choices[0].message.content.strip())
    except Exception:
        return 0

# ================= RUTAS =================
@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()

    cur.execute("""
        SELECT 
            m.id,
            t.name,
            m.points,
            m.created_at,
            m.details,
            m.image_path
        FROM matches m
        JOIN teams t ON t.id = m.team_id
        ORDER BY m.created_at DESC
    """)
    
    raw_matches = cur.fetchall()
    matches = []

    for m in raw_matches:
        details = {}
        if m[4]:
            try:
                details = json.loads(m[4])
            except:
                details = {}

        matches.append((
            m[0],  # id
            m[1],  # team name
            m[2],  # points
            m[3],  # created_at
            details,  # ✅ dict
            m[5],  # image_path
        ))

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

    ext = image.filename.rsplit(".", 1)[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    points = calcular_puntos_domino(image_path)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path)
        VALUES (%s, %s, %s)
    """, (team_id, points, image_path))

    cur.execute("""
        UPDATE teams
        SET points = points + %s
        WHERE id = %s
    """, (points, team_id))

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





