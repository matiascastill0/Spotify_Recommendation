# SPOTIFYRECOMMENDATION/data/raw/download_kaggle.py

import os

print("🚀 Descargando dataset de Spotify desde Kaggle...")

DATASET = "zinasakr/40k-songs-with-audio-features-and-lyrics"

# Asegúrate de tener configurado tu kaggle.json en ~/.kaggle/
os.system(f"kaggle datasets download -d {DATASET} -p ./data/raw/ --unzip")

print("✅ Descarga completada. Dataset guardado en data/raw/")
