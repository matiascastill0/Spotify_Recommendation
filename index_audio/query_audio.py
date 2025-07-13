# SPOTIFYRECOMMENDATION/index_audio/query_audio.py

import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# CONFIG
DATA_PATH = "./data/raw/final_dataset.csv"
CODEBOOK_PATH = "./index_audio/codebook.pkl"
HISTOGRAMS_PATH = "./index_audio/histograms.pkl"
N_CLUSTERS = 50
TOP_K = 5

# 1️⃣ Cargar dataset limpio
df = pd.read_csv(DATA_PATH)

# 2️⃣ Extraer columnas de audio features
features = df[['acousticness', 'danceability', 'energy', 'instrumentalness',
               'liveness', 'loudness', 'speechiness', 'tempo', 'valence']].fillna(0).values

# 3️⃣ Usar la primera fila como query
query_vector = features[0].reshape(1, -1)

# 4️⃣ Cargar codebook y histogramas
with open(CODEBOOK_PATH, "rb") as f:
    codebook = pickle.load(f)

with open(HISTOGRAMS_PATH, "rb") as f:
    histograms = pickle.load(f)

# 5️⃣ KMeans para predecir cluster usando codebook
# Calcular distancia a cada centroide del codebook
distances = np.linalg.norm(codebook - query_vector, axis=1)
cluster = np.argmin(distances)


# 6️⃣ Crear histograma query (1-hot)
query_hist = np.zeros(N_CLUSTERS)
query_hist[cluster] = 1

# 7️⃣ Calcular similitud coseno con todos los histogramas
results = []
for idx, hist in histograms.items():
    sim = cosine_similarity([query_hist], [hist])[0][0]
    results.append((idx, sim))

results.sort(key=lambda x: x[1], reverse=True)

# 8️⃣ Mostrar resultados con metadatos
print(f"\n🎧 Top-{TOP_K} canciones más similares:")

for idx, score in results[:TOP_K]:
    fila = df.iloc[idx]
    print(f"{fila['artist_name']} - {fila['song_name']} | Score: {score:.4f}")
