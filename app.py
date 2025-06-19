from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "¡Hola, Mundo desde Flask!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict(np.array(data["features"]).reshape(1, -1))
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)