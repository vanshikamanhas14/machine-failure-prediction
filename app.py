from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
try:
    model = joblib.load("machine_failure_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print("Model loading error:", e)


@app.route("/")
def home():
    return "Machine Failure Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]

        prediction = model.predict([data])[0]

        if prediction == 1:
            prediction_text = "Machine Failure"

            explanation = (
                "The machine is likely to fail because the sensor readings "
                "indicate abnormal operating conditions. High temperature, "
                "excessive torque, abnormal vibration, or increased tool wear "
                "can contribute to machine failure."
            )

        else:
            prediction_text = "No Machine Failure"

            explanation = (
                "The machine is operating under normal conditions. "
                "No significant indicators of machine failure were detected "
                "from the provided sensor readings."
            )

        return jsonify({
            "prediction": prediction_text,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)