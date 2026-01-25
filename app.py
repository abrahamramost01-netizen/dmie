import os
import uuid
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# =========================
# CONFIG
# =========================
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model.onnx"
IMG_SIZE = 640
CONF_THRES = 0.35
IOU_THRES = 0.45

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# LOAD ONNX MODEL
# =========================
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# =========================
# UTILS
# =========================
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def non_max_merge(boxes, scores, classes):
    final = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        group = [i]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if iou(boxes[i], boxes[j]) > IOU_THRES:
                used[j] = True
                group.append(j)

        best = max(group, key=lambda k: scores[k])
        final.append(best)

    return final


# =========================
# DOMINO LOGIC
# =========================
def calcular_puntos_domino(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img, 1 / 255.0, (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward()[0]

    boxes, scores, classes = [], [], []

    for det in outputs:
        conf = float(det[4])
        if conf < CONF_THRES:
            continue

        cls_scores = det[5:]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id] * conf)

        cx, cy, bw, bh = det[:4]
        x = int((cx - bw / 2) * w / IMG_SIZE)
        y = int((cy - bh / 2) * h / IMG_SIZE)
        bw = int(bw * w / IMG_SIZE)
        bh = int(bh * h / IMG_SIZE)

        boxes.append([x, y, bw, bh])
        scores.append(score)
        classes.append(cls_id)

    idxs = non_max_merge(boxes, scores, classes)

    fichas = []
    total = 0

    for i in idxs:
        clase = classes[i]
        # clase = 0-66 → dominó doble-nueve
        valor = clase % 10 + clase // 10

        confianza = round(scores[i] * 100, 2)
        confianza = min(confianza, 100.0)

        total += valor

        fichas.append({
            "clase": clase,
            "puntos": valor,
            "confianza": confianza,
            "box": boxes[i]
        })

        x, y, bw, bh = boxes[i]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{valor} ({confianza}%)",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    out_path = image_path.replace(".jpg", "_det.jpg")
    cv2.imwrite(out_path, img)

    return {
        "total": total,
        "cantidad": len(fichas),
        "fichas": fichas,
        "imagen": os.path.basename(out_path),
    }


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            name = f"{uuid.uuid4()}.jpg"
            path = os.path.join(app.config["UPLOAD_FOLDER"], name)
            file.save(path)

            resultado = calcular_puntos_domino(path)

    return render_template("index.html", resultado=resultado)


@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


