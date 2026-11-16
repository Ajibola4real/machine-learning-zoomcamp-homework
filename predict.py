
import pickle
import pandas as pd
from flask import Flask, request, jsonify

with open("employee_attrition_model_from_script.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/")
def home():
    return "Employee Attrition Prediction API is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    employee = request.get_json()
    employee_df = pd.DataFrame([employee])

    probability = float(model.predict_proba(employee_df)[0, 1])
    attrition = bool(probability >= 0.5)

    return jsonify({
        "attrition_probability": probability,
        "attrition": attrition
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
