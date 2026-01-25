from flask import Flask, request, jsonify
import numpy as np
import cv2
import onnxruntime as ort

app = Flask(__name__)

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name


@app.route("/add_match", methods=["POST"])
def add_match():
    file = request.files["image"]
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.transpose(2, 0, 1)
    img_input = img_input[np.newaxis, :].astype(np.float32) / 255.0

    output = session.run(None, {input_name: img_input})[0]

    return jsonify({
        "input_shape": list(img_input.shape),
        "output_shape": list(output.shape),
        "raw_output_sample": output[0][0][:50].tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)

