from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

category_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Science/Technology"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    news_vector = vectorizer.transform([news])

    result = model.predict(news_vector)

    prediction = category_map[int(result[0])]

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)