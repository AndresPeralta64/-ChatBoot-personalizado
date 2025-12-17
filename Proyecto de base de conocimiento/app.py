from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# --- Configuration ---
DATASET_FILE = "dataset mecánica cuántica.csv"
SIMILARITY_THRESHOLD = 0.2

# --- Chatbot Logic ---
class QuantumBot:
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        self.df = pd.read_csv(dataset_path)
        # Combine title and text for better matching
        self.df['combined_text'] = self.df['titulo'] + " " + self.df['texto']
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'].fillna(''))

    def get_response(self, user_input):
        user_tfidf = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        best_match_index = similarities.argmax()
        best_score = similarities[best_match_index]

        if best_score > SIMILARITY_THRESHOLD:
            return self.df.iloc[best_match_index]['texto']
        else:
            return "Lo siento, mi conocimiento se limita a la mecánica cuántica y no encuentro una respuesta en mi base de datos para esa pregunta."

# Initialize Bot
try:
    bot = QuantumBot(DATASET_FILE)
except Exception as e:
    print(f"Error initializing bot: {e}")
    bot = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if not bot:
        return jsonify({'response': "Error: El dataset no se pudo cargar."})
    
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'response': "Por favor escribe algo."})
    
    response = bot.get_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
