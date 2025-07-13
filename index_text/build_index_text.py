# build_index_text.py

import os
import re
import math
import pickle
import nltk
import pandas as pd
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# CONFIG
DATA_PATH = "./data/raw/final_dataset.csv"
OUTPUT_INDEX = "./index_text/index_inverted.pkl"

# Inicializar
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Leer dataset
df = pd.read_csv(DATA_PATH)
documents = []
for _, row in df.iterrows():
    text = f"{row['song_name']} {row['artist_name']} {row['lyrics']}"
    documents.append(text)

# Preprocesamiento
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    stems = [stemmer.stem(w) for w in tokens]
    return stems

# Construir índice invertido
inverted_index = defaultdict(list)
doc_norms = {}

N = len(documents)
df_counts = defaultdict(int)

# Primer paso: calcular DF
for doc_id, text in enumerate(documents):
    terms = set(preprocess(text))
    for term in terms:
        df_counts[term] += 1

# Segundo paso: TF-IDF y SPIMI simplificado
for doc_id, text in enumerate(documents):
    tokens = preprocess(text)
    tf = defaultdict(int)
    for term in tokens:
        tf[term] += 1

    norm = 0
    for term, freq in tf.items():
        tf_idf = freq * math.log(N / (1 + df_counts[term]))
        inverted_index[term].append((doc_id, tf_idf))
        norm += tf_idf ** 2

    doc_norms[doc_id] = math.sqrt(norm)

# Guardar índice invertido y normas
os.makedirs("./index_text/", exist_ok=True)
with open(OUTPUT_INDEX, "wb") as f:
    pickle.dump({
        "inverted_index": inverted_index,
        "doc_norms": doc_norms,
        "df_counts": df_counts,
        "N": N
    }, f)

print(f"✅ Índice invertido guardado en {OUTPUT_INDEX}")
