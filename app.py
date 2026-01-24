import os
import json
import base64
import psycopg2
from flask import Flask, request, redirect, render_template_string
from openai import OpenAI

# ================= CONFIG =================

app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DATABASE_URL = os.environ["DATABASE_URL"]

# ================= DB =================

def get_db():
    return psycopg2.connect(DATABASE_URL)

# ================= IA =================

PROMPT = """
Eres un sistema experto en dominó DOBLE 9.

Tareas:
1. Detecta TODAS las fichas visibles aunque estén juntas o tocándose.
2. Cada ficha tiene dos valores (0 a 9).
3. Devuelve TODAS las fichas detectadas.
4. Calcula el total sumando TODOS los lados.
5. Incluye bounding boxes aproximadas.

Devuelve SOLO JSON con esta estructura exacta:

{
  "total": number,
  "fichas": [
    {
      "valores": [int, int],
      "suma": int,
      "box": { "x": int, "y": int, "w": int, "h": int }
    }
  ]
}
"""

def calcular_puntos_domino(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{img_b64}"
                }
            ]
        }],
        max_output_tokens=500
    )

    text = response.output_text
    return json.loads(text)

# ================= ROUTES =================

@app.route("/", methods=["GET"])
def index():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, points, details, image_path
        FROM matches
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return render_template_string(TEMPLATE, matches=rows)

@app.route("/add_match", methods=["POST"])
def add_match():
    image = request.files["image"]
    path = f"uploads/{image.filename}"
    os.makedirs("uploads", exist_ok=True)
    image.save(path)

    resultado = calcular_puntos_domino(path)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO matches (points, details, image_path)
        VALUES (%s, %s, %s)
    """, (
        resultado["total"],
        json.dumps(resultado),
        path
    ))
    conn.commit()
    cur.close()
    conn.close()

    return redirect("/")

# ================= HTML =================

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Dominó IA</title>
<style>
.box {
  position:absolute;
  border:2px solid red;
}
.container {
  position:relative;
  display:inline-block;
}
</style>
</head>
<body>

<h2>Subir partida</h2>
<form method="POST" action="/add_match" enctype="multipart/form-data">
<input type="file" name="image" required>
<button>Enviar</button>
</form>

<hr>

{% for id, points, details, image in matches %}
<h3>Total: {{ points }}</h3>

<div class="container">
<img src="{{ image }}" width="400">

{% set d = details | tojson | safe %}
{% for f in details.fichas %}
<div class="box"
style="
left:{{f.box.x}}px;
top:{{f.box.y}}px;
width:{{f.box.w}}px;
height:{{f.box.h}}px;">
</div>
{% endfor %}
</div>

<ul>
{% for f in details.fichas %}
<li>{{ f.valores[0] }} + {{ f.valores[1] }} = {{ f.suma }}</li>
{% endfor %}
</ul>

<hr>
{% endfor %}

</body>
</html>
"""

# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)




