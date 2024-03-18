import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)


def load_model():
    model = pickle.load(open(r"backend\artifacts\model.pkl", "rb"))
    return model


@app.route("/")
def home():
    data = {
        'title': 'Welcome to breast cancer classification app',
        'message': 'This is an example integration between Angular and Python!'
    }
    return jsonify(data)


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.asarray(int_features)]
    predicted_class = load_model().predict(final_features)
    print(int_features)
    prediction = ""
    if predicted_class == 0:
        prediction = prediction + "Benign Cancer"
    else:
        prediction = prediction + "Malignant Cancer"

    return render_template(
        "index.html", prediction_text=f"You're breast cancer is a {prediction}"
    )


if __name__ == "__main__":
    app.run(debug=True)
