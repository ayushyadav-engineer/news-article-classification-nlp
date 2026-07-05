# News Article Classification Using NLP

A Natural Language Processing (NLP) web application that classifies news articles into predefined categories using Machine Learning. The project preprocesses textual data, trains a classification model, and provides predictions through an interactive Flask-based web interface.

---

## Features

* News article classification using Machine Learning
* Text preprocessing and cleaning
* TF-IDF vectorization
* Model training and evaluation
* Real-time prediction through a web interface
* Pre-trained model for instant predictions
* Simple and responsive user interface

---

## Technologies Used

* Python 3
* Flask
* Scikit-learn
* Pandas
* NumPy
* NLTK
* HTML5
* CSS3

---

## Project Structure

```text
News-Classification/
│
├── dataset/
│   └── train.csv
│
├── model/
│   ├── news_model.pkl
│   └── vectorizer.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── preprocess.py
├── train_model.py
├── app.py
├── requirements.txt
│
└── accuracy.png
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/news-article-classification-nlp.git
```

### Navigate to the project directory

```bash
cd news-article-classification-nlp
```

### Create a virtual environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install the required packages

```bash
pip install -r requirements.txt
```

### Train the model (Optional)

If you want to generate a new trained model from the dataset:

```bash
python train_model.py
```

This creates:

* `news_model.pkl`
* `vectorizer.pkl`

inside the `model/` directory.

### Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Workflow

1. Load the news dataset.
2. Preprocess and clean the text.
3. Convert text into numerical features using TF-IDF.
4. Train the Machine Learning model.
5. Save the trained model and vectorizer.
6. Launch the Flask application.
7. Enter a news article and receive its predicted category.

---

## Model Performance

The project includes an accuracy graph:

```
accuracy.png
```

This image illustrates the classification accuracy achieved by the trained model.

---

## Future Improvements

* Support multiple Machine Learning algorithms
* Integrate BERT or Transformer-based models
* Add confidence scores for predictions
* Deploy on cloud platforms
* REST API integration
* User authentication and prediction history

---

## License

This project is developed for educational and academic purposes.

---

## Author

**Ayush Yadav**
