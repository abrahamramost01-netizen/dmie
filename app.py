from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)

MODEL_PATH = "best.onnx"
OUTPUT_DIR = "static/out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# cargar modelo SOLO una vez
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]

    # preprocessing YOLO típico
    input_img = cv2.resize(img, (640, 640))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_img})[0]

    detections = []

    for det in output[0]:
        conf = float(det[4])
        if conf < 0.4:
            continue

        cls = int(det[5])

        # 👉 ASUMIMOS QUE YA VIENEN EN PIXELES
        x1, y1, x2, y2 = map(int, det[:4])

        detections.append({
            "box": [x1, y1, x2, y2],
            "clase": cls,
            "conf": conf
        })

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    name = f"proc_{uuid.uuid4()}.jpg"
    out_path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(out_path, img)

    return jsonify({
        "detections": detections,
        "image": name
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)





