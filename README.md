# Text Similarity Calculator

This Flask web application calculates the semantic similarity between two input paragraphs using the Universal Sentence Encoder (USE) model.

## Features

- Allows users to input two paragraphs and calculates their semantic similarity.
- Cleans the input text by removing HTML tags, non-alphanumeric characters, and extra whitespaces.
- Utilizes the Universal Sentence Encoder (USE) model from TensorFlow Hub to encode the input paragraphs into dense vectors.
- Calculates cosine similarity between the embeddings of the input paragraphs to determine their semantic similarity.
- Provides a user-friendly web interface for interacting with the application.


## Prerequisites

- Python 3.x
- Flask
- scikit-learn
- BeautifulSoup4
- TensorFlow Hub
- TensorFlow (optional)

