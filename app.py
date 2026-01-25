import os
import uuid
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")
IMG_SIZE = 640

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# ================= LOAD ONNX =================
session = None
input_name = None
output_name = None

print(f"🔧 Intentando cargar ONNX: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    try:
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print("✅ ONNX cargado correctamente")
    except Exception as e:
        print("❌ Error cargando ONNX:", e)
else:
    print("❌ El archivo ONNX no existe")

# ================= UTILS =================
def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("No se pudo leer la imagen")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/raw_predict", methods=["POST"])
def raw_predict():
    if session is None:
        return jsonify({"error": "Modelo ONNX no cargado"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No se envió imagen"}), 400

    image = request.files["image"]
    ext = os.path.splitext(image.filename)[1].lower()
    name = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_FOLDER, name)
    image.save(path)

    try:
        img_input = preprocess(path)
        outputs = session.run([output_name], {input_name: img_input})

        # DEVOLVEMOS TODO SIN TOCAR
        return jsonify({
            "image": name,
            "output_shape": np.array(outputs[0]).shape,
            "raw_output": outputs[0].tolist()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/health")
def health():
    return {
        "status": "ok",
        "onnx_loaded": session is not None
    }

# ================= START =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


