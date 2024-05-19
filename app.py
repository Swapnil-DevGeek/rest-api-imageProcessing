from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import numpy as np
from model import model, preprocess_input, decode_predictions
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image, target_size=(224, 224))

        predictions = model.predict(processed_image)
        results = decode_predictions(predictions, top=3)[0]

        response = []
        for (imagenet_id, label, score) in results:
            response.append({
                "label": label,
                "score": float(score)
            })

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
