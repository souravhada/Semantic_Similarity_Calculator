from flask import Flask, render_template, request, redirect
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import re
from bs4 import BeautifulSoup

MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
app = Flask(__name__)

# Step 2: Clean the text
def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(' +', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Step 3: Load the USE model
def load_model():
    use_model_url = MODEL_URL
    use_model = hub.load(use_model_url)
    return use_model

# Step 4: Calculate cosine similarity
def calculate_similarity(text1, text2):
    # Clean the input paragraphs
    cleantext1 = clean_text(text1)
    cleantext2 = clean_text(text2)

    use_model = load_model()
    # Encode the paragraphs
    embeddings_text1 = use_model([cleantext1])
    embeddings_text2 = use_model([cleantext2])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(embeddings_text1, embeddings_text2)

    return similarity_scores[0][0]

# Define the route for the home page where the user can input the paragraphs
@app.route('/', methods=['GET', 'POST'])
def home():
    similarity_score = None
    text1 = ""
    text2 = ""
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        similarity_score = calculate_similarity(text1, text2)

    return render_template('index.html', similarity_score=similarity_score, text1=text1, text2=text2)  # Pass text1 and text2 as variables

# Define the route for calculating the similarity
@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity_route():

    text1 = request.form['text1']
    text2 = request.form['text2']
    similarity_score = calculate_similarity(text1, text2)
    return render_template('index.html', similarity_score=similarity_score, text1=text1, text2=text2)  # Pass text1 and text2 as variables

# Define the route for resetting the paragraphs
@app.route('/reset', methods=['GET'])
def reset():
    return redirect('/')  # Redirect to the home page

# Run the Flask app
if __name__ == '__main__':
    # Set up the Flask app
    app.run(debug=True)


