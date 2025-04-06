from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import fitz 
import io

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("document_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Get category names
categories = fetch_20newsgroups(subset='train').target_names

@app.route('/', methods=["GET", "POST"])
def index():
    top_predictions = None
    user_input = ""

    if request.method == "POST":
        uploaded_file = request.files.get('file')

        if uploaded_file:
            filename = uploaded_file.filename.lower()

            if filename.endswith('.txt'):
                user_input = uploaded_file.read().decode("utf-8")

            elif filename.endswith('.pdf'):
                pdf_data = uploaded_file.read()
                pdf_doc = fitz.open(stream=pdf_data, filetype="pdf")
                text = ""
                for page in pdf_doc:
                    text += page.get_text()
                user_input = text

        else:
            user_input = request.form.get('text', '')

        if user_input.strip():
            input_vector = vectorizer.transform([user_input])
            decision_scores = model.decision_function(input_vector)

            if decision_scores.ndim == 1:
                pred_index = np.argmax(decision_scores)
                top_predictions = [(categories[pred_index], 100.0)]
            else:
                top3_indices = np.argsort(decision_scores[0])[::-1][:3]
                top3_scores = decision_scores[0][top3_indices]
                top3_probs = np.exp(top3_scores) / np.sum(np.exp(top3_scores))  # softmax

                top_predictions = [(categories[i], round(top3_probs[j]*100, 2)) for j, i in enumerate(top3_indices)]

    return render_template("index.html", predictions=top_predictions, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
