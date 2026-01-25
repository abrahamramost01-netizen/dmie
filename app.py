import os
import uuid
import json
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
WIN_POINTS = int(os.environ.get("WIN_POINTS", 200))
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
DATABASE_URL = os.environ.get("DATABASE_URL")

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD YOLO =================
print(f"🔧 Loading YOLO model: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLO model loaded")
except Exception as e:
    print(f"❌ YOLO load failed: {e}")
    model = None

# ================= DB =================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= YOLO LOGIC =================
def calcular_puntos_domino(image_path):
    if model is None:
        return {"total": 0, "cantidad": 0, "fichas": []}

    results = model(image_path, conf=0.5, verbose=False)

    total = 0
    fichas = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            puntos = cls  # Ajusta si tu dataset usa otro mapping

            fichas.append({
                "puntos": puntos,
                "confianza": round(conf * 100, 1)
            })
            total += puntos

    return {
        "total": total,
        "cantidad": len(fichas),
        "fichas": fichas
    }

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
        ORDER BY m.id DESC
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

    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        return redirect(url_for("index"))

    filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    resultado = calcular_puntos_domino(path)
    points = resultado["total"]
    details_json = json.dumps(resultado)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO matches (team_id, points, image_path, details)
        VALUES (%s, %s, %s, %s)
    """, (team_id, points, path, details_json))

    cur.execute("""
        UPDATE teams SET points = points + %s WHERE id = %s
    """, (points, team_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

