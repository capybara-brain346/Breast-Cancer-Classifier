from flask import Flask, render_template, url_for, request
import pickle

app = Flask(__name__)
model = pickle.load(open("../model/random_forest_model.pkl"))


@app.route('/')
def landing_page():
    return render_template("index.html")


if __name__=="__main__":
    app.run(debug=True)
