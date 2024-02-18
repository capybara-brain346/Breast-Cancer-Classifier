import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open(r'model\random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.asarray(int_features)]
    predicted_class = model.predict(final_features)
    prediction = ""
    if predicted_class==0:
        prediction = prediction+"Benign Cancer"
    else: 
        prediction = prediction+"Malignant Cancer"
        
    return render_template('index.html', prediction_text=f"You're breast cancer is a {prediction}")


if __name__ == "__main__":
    app.run(debug=True)