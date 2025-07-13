# query_text.py

import math
import pickle
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    import re
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    stems = [stemmer.stem(w) for w in tokens]
    return stems

# Cargar índice invertido
with open("./index_text/index_inverted.pkl", "rb") as f:
    data = pickle.load(f)

index = data['inverted_index']
doc_norms = data['doc_norms']
df_counts = data['df_counts']
N = data['N']

# Consulta
query = "love war time"
query_terms = preprocess(query)

# Calcular pesos consulta
query_tf = defaultdict(int)
for term in query_terms:
    query_tf[term] += 1

query_vec = {}
for term, freq in query_tf.items():
    idf = math.log(N / (1 + df_counts.get(term, 1)))
    query_vec[term] = freq * idf

# Scoring por similitud de coseno
scores = defaultdict(float)
for term, q_weight in query_vec.items():
    for doc_id, d_weight in index.get(term, []):
        scores[doc_id] += q_weight * d_weight

for doc_id in scores:
    scores[doc_id] /= (doc_norms[doc_id] or 1)

# Top-K
top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top-K resultados:")
for doc_id, score in top_k:
    print(f"DocID: {doc_id} | Score: {score:.4f}")
