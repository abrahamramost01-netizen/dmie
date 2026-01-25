import os
import uuid
import json
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, send_from_directory

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")
IMG_SIZE = 640

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# ================= LOAD MODEL =================
print("🔧 Cargando modelo ONNX...")
try:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print("✅ Modelo cargado")
    print("➡ Inputs:", input_name)
    print("➡ Outputs:", output_names)
except Exception as e:
    print("❌ Error cargando modelo:", e)
    session = None

# ================= PREPROCESS =================
def preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ================= ROUTES =================
@app.route("/")
def home():
    return """
    <h2>Debug YOLO ONNX</h2>
    <form action="/debug_raw" method="post" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Subir imagen</button>
    </form>
    """

@app.route("/debug_raw", methods=["POST"])
def debug_raw():
    if session is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image"}), 400

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    img = preprocess(path)

    outputs = session.run(None, {input_name: img})

    # 🔥 DEVOLVEMOS TODO TAL CUAL
    raw = []
    for i, out in enumerate(outputs):
        raw.append({
            "output_index": i,
            "shape": list(out.shape),
            "sample": out[0][:5].tolist() if out.ndim >= 2 else out.tolist()
        })

    return jsonify({
        "image": filename,
        "outputs_count": len(outputs),
        "raw_outputs": raw
    })

@app.route("/uploads/<name>")
def uploads(name):
    return send_from_directory(UPLOAD_FOLDER, name)

@app.route("/health")
def health():
    return {
        "status": "ok",
        "onnx_loaded": session is not None
    }

# ================= START =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

