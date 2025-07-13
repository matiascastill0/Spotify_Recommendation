# SPOTIFYRECOMMENDATION/data/raw/eda_kaggle.py

import pandas as pd

# Ruta del CSV descargado
CSV_PATH = "./data/raw/final_dataset.csv"  # Ajusta si cambia el nombre

# Cargar dataset
df = pd.read_csv(CSV_PATH)

print("🔍 Shape del dataset:", df.shape)
print("\n📋 Columnas disponibles:\n", df.columns)

# Verificar algunos datos clave
print("\n📌 Ejemplo de filas:")
print(df.head())

print("\n❓ Filas con letras vacías:")
print(df['lyrics'].isnull().sum())

print("\n❓ Filas con audio features nulos:")
print(df.isnull().sum())

# Opcional: filtrar canciones con letra
df_clean = df[df['lyrics'].notnull()]
print("\n✅ Canciones con letra disponible:", df_clean.shape)

# Guardar versión filtrada
df_clean.to_csv("./data/raw/final_dataset.csv", index=False)
print("\n✅ Archivo filtrado guardado como final_dataset.csv")
