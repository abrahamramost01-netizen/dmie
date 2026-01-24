import os
import uuid
import json
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
import cv2

# ====== CONFIGURACIÓN ======
UPLOAD_FOLDER = "uploads"
WIN_POINTS = int(os.environ.get("WIN_POINTS", 200))
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ====== CARGAR MODELO YOLO ======
print(f"🔧 Cargando modelo YOLO desde: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Modelo YOLO cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None

# ====== BASE DE DATOS ======
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    """Conexión a PostgreSQL"""
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ====== LÓGICA DE DETECCIÓN ======
def calcular_puntos_domino(image_path):
    """
    Detecta fichas de dominó usando YOLO y calcula puntos totales.
    
    Retorna:
        dict: {
            "total": int,
            "fichas": [{"clase": int, "puntos": int, "confianza": float}],
            "cantidad": int
        }
    """
    if model is None:
        print("⚠️ Modelo no disponible, retornando 0 puntos")
        return {"total": 0, "fichas": [], "cantidad": 0, "error": "Modelo no cargado"}
    
    try:
        # Ejecutar detección
        results = model(image_path, conf=0.5, verbose=False)
        
        fichas = []
        total = 0
        
        # Procesar cada detección
        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                
                # En dominó, la clase representa los puntos de la ficha
                # Ajusta esto según tu esquema de etiquetado
                # Ejemplo: clase 0 = blanca (0 puntos), clase 1 = 1 punto, etc.
                puntos = cls  # O usa un mapeo personalizado si es diferente
                
                fichas.append({
                    "clase": cls,
                    "puntos": puntos,
                    "confianza": round(conf * 100, 1)
                })
                
                total += puntos
        
        resultado = {
            "total": total,
            "fichas": fichas,
            "cantidad": len(fichas)
        }
        
        print(f"✅ Detección exitosa: {len(fichas)} fichas, {total} puntos")
        return resultado
        
    except Exception as e:
        print(f"❌ Error en detección: {e}")
        return {"total": 0, "fichas": [], "cantidad": 0, "error": str(e)}

# ====== RUTAS ======
@app.route("/")
def index():
    """Página principal"""
    conn = get_db()
    cur = conn.cursor()
    
    # Obtener equipos
    cur.execute("SELECT id, name, points FROM teams ORDER BY id")
    teams = cur.fetchall()
    
    # Obtener partidas con detalles
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
    """Crear nuevo equipo"""
    name = request.form.get("name", "").strip()
    
    if not name:
        return redirect(url_for("index"))
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        cur.execute("INSERT INTO teams (name) VALUES (%s)", (name,))
        conn.commit()
        print(f"✅ Equipo creado: {name}")
    except Exception as e:
        print(f"❌ Error creando equipo: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
    
    return redirect(url_for("index"))

@app.route("/edit_team_points", methods=["POST"])
def edit_team_points():
    """Editar puntos de un equipo"""
    try:
        team_id = int(request.form.get("team_id"))
        points = int(request.form.get("points"))
        
        conn = get_db()
        cur = conn.cursor()
        
        cur.execute("UPDATE teams SET points = %s WHERE id = %s", (points, team_id))
        conn.commit()
        
        print(f"✅ Puntos actualizados: Equipo {team_id} → {points}")
        
        cur.close()
        conn.close()
        
    except (ValueError, TypeError) as e:
        print(f"❌ Error en datos: {e}")
    except Exception as e:
        print(f"❌ Error actualizando puntos: {e}")
    
    return redirect(url_for("index"))

@app.route("/delete_team", methods=["POST"])
def delete_team():
    """Eliminar equipo"""
    try:
        team_id = int(request.form.get("team_id"))
        
        conn = get_db()
        cur = conn.cursor()
        
        cur.execute("DELETE FROM teams WHERE id = %s", (team_id,))
        conn.commit()
        
        print(f"✅ Equipo eliminado: {team_id}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error eliminando equipo: {e}")
    
    return redirect(url_for("index"))

@app.route("/add_match", methods=["POST"])
def add_match():
    """Registrar nueva partida con detección YOLO"""
    try:
        team_id = int(request.form.get("team_id"))
        image = request.files.get("image")
        
        if not image:
            print("⚠️ No se recibió imagen")
            return redirect(url_for("index"))
        
        # Validar tipo de archivo
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = os.path.splitext(image.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            print(f"⚠️ Extensión no permitida: {file_ext}")
            return redirect(url_for("index"))
        
        # Guardar imagen
        filename = f"{uuid.uuid4()}{file_ext}"
        path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(path)
        
        print(f"📸 Imagen guardada: {path}")
        
        # Calcular puntos con YOLO
        resultado = calcular_puntos_domino(path)
        points = resultado["total"]
        details_json = json.dumps(resultado)
        
        # Guardar en BD
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
        
        print(f"✅ Partida registrada: {points} puntos para equipo {team_id}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error en add_match: {e}")
    
    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploads(filename):
    """Servir archivos subidos"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/health")
def health():
    """Health check para Railway"""
    status = {
        "status": "ok",
        "model_loaded": model is not None,
        "database": "connected"
    }
    
    try:
        conn = get_db()
        conn.close()
    except:
        status["database"] = "error"
        status["status"] = "degraded"
    
    return status

# ====== INICIALIZACIÓN ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    print(f"🚀 Iniciando Flask en puerto {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
