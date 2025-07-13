# SPOTIFYRECOMMENDATION/frontend/app.py

from flask import Flask, render_template, request
import pickle
import math
import re
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# CONFIGS
INDEX_TEXT_PATH = "../index_text/index_inverted.pkl"
CODEBOOK_PATH = "../index_audio/codebook.pkl"
HISTOGRAMS_PATH = "../index_audio/histograms.pkl"
DATA_PATH = "../data/raw/final_dataset.csv"
N_CLUSTERS = 50

# NLP
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    stems = [stemmer.stem(w) for w in tokens]
    return stems

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search_text", methods=["POST"])
def search_text():
    query = request.form["query"]
    k = int(request.form["k"])

    # Cargar índice invertido
    with open(INDEX_TEXT_PATH, "rb") as f:
        data = pickle.load(f)

    index = data['inverted_index']
    doc_norms = data['doc_norms']
    df_counts = data['df_counts']
    N = data['N']

    query_terms = preprocess(query)
    query_tf = defaultdict(int)
    for term in query_terms:
        query_tf[term] += 1

    query_vec = {}
    for term, freq in query_tf.items():
        idf = math.log(N / (1 + df_counts.get(term, 1)))
        query_vec[term] = freq * idf

    scores = defaultdict(float)
    for term, q_weight in query_vec.items():
        for doc_id, d_weight in index.get(term, []):
            scores[doc_id] += q_weight * d_weight

    for doc_id in scores:
        scores[doc_id] /= (doc_norms[doc_id] or 1)

    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    # Obtener metadatos
    df = pd.read_csv(DATA_PATH)
    results_full = []
    for doc_id, score in results:
        fila = df.iloc[doc_id]
        results_full.append({
            'artist': fila['artist_name'],
            'song': fila['song_name'],
            'genres': fila['genres'],
            'popularity': fila['new_artist_popularity'],
            'score': round(score, 4)
        })

    return render_template("index.html", results=results_full)

@app.route("/search_audio", methods=["POST"])
def search_audio():
    query_row = int(request.form["query_row"])

    df = pd.read_csv(DATA_PATH)
    features = df[['acousticness', 'danceability', 'energy', 'instrumentalness',
                   'liveness', 'loudness', 'speechiness', 'tempo', 'valence']].fillna(0).values

    query_vector = features[query_row].reshape(1, -1)

    with open(CODEBOOK_PATH, "rb") as f:
        codebook = pickle.load(f)

    with open(HISTOGRAMS_PATH, "rb") as f:
        histograms = pickle.load(f)

    # Asignar cluster (distancia euclidiana)
    distances = np.linalg.norm(codebook - query_vector, axis=1)
    cluster = np.argmin(distances)

    # Crear histograma query (1-hot)
    query_hist = np.zeros(N_CLUSTERS)
    query_hist[cluster] = 1

    results = []
    for idx, hist in histograms.items():
        sim = cosine_similarity([query_hist], [hist])[0][0]
        results.append((idx, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    top_results = []
    for idx, score in results[:5]:
        fila = df.iloc[idx]
        top_results.append({
            'artist': fila['artist_name'],
            'song': fila['song_name'],
            'genres': fila['genres'],
            'popularity': fila['new_artist_popularity'],
            'score': round(score, 4)
        })

    return render_template("index.html", audio_results=top_results)

if __name__ == "__main__":
    app.run(debug=True)

