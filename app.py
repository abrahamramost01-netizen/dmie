import os
import uuid
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template, send_from_directory

# ================== CONFIG ==================
UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")
IMG_SIZE = 640

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== LOAD ONNX ==================
print(f"Cargando modelo ONNX: {MODEL_PATH}")
try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Modelo ONNX cargado OK")
except Exception as e:
    print("ERROR cargando ONNX:", e)
    session = None

# ================== UTILS ==================
def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ================== ROUTES ==================
@app.route("/")
def index():
    return render_template("raw.html")

@app.route("/raw_detect", methods=["POST"])
def raw_detect():
    if session is None:
        return jsonify({"error": "Modelo no cargado"})

    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image"})

    ext = os.path.splitext(image.filename)[1]
    name = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_FOLDER, name)
    image.save(path)

    img_input = preprocess(path)

    # OUTPUT CRUDO
    outputs = session.run([output_name], {input_name: img_input})

    return jsonify({
        "input_shape": img_input.shape,
        "output_type": str(type(outputs)),
        "output_shape": np.array(outputs[0]).shape,
        "raw_output_sample": outputs[0][0][:10].tolist()  # primeros 10
    })

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/health")
def health():
    return {"status": "ok", "onnx": session is not None}

# ================== START ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
