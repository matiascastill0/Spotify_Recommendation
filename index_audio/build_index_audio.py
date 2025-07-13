# SPOTIFYRECOMMENDATION/index_audio/build_index_audio.py

import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

# CONFIG
DATA_PATH = "./data/raw/final_dataset.csv"
OUTPUT_CODEBOOK = "./index_audio/codebook.pkl"
OUTPUT_HISTOGRAMS = "./index_audio/histograms.pkl"
N_CLUSTERS = 50

# Leer dataset limpio
df = pd.read_csv(DATA_PATH)

# Seleccionar columnas de audio features
features = df[['acousticness', 'danceability', 'energy', 'instrumentalness',
               'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]

# Rellenar valores faltantes
X = features.fillna(0).values

print(f"🚀 Vector shape: {X.shape}")

# K-Means clustering
print("🚀 Entrenando K-Means...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
kmeans.fit(X)

# Codebook = centros de cluster
codebook = kmeans.cluster_centers_

# Generar histogramas simulados
histograms = {}
clusters = kmeans.predict(X)
for idx, label in enumerate(clusters):
    hist = np.zeros(N_CLUSTERS)
    hist[label] = 1
    histograms[idx] = hist

# Guardar
with open(OUTPUT_CODEBOOK, "wb") as f:
    pickle.dump(codebook, f)

with open(OUTPUT_HISTOGRAMS, "wb") as f:
    pickle.dump(histograms, f)

print(f"✅ Codebook guardado en {OUTPUT_CODEBOOK}")
print(f"✅ Histogramas guardados en {OUTPUT_HISTOGRAMS}")
