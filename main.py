# SPOTIFYRECOMMENDATION/main.py

from index_text import build_index_text, query_text
from index_audio import build_index_audio, query_audio

if __name__ == "__main__":
    print("🗂️ Opciones:")
    print("1: Construir índice invertido SPIMI")
    print("2: Consultar texto")
    print("3: Construir índice audio (MFCC + KMeans)")
    print("4: Consultar audio")
    print("5: Lanzar Flask")

    opcion = input("Selecciona opción (1-5): ")

    if opcion == "1":
        exec(open("index_text/build_index_text.py").read())
    elif opcion == "2":
        exec(open("index_text/query_text.py").read())
    elif opcion == "3":
        exec(open("index_audio/build_index_audio.py").read())
    elif opcion == "4":
        exec(open("index_audio/query_audio.py").read())
    elif opcion == "5":
        from frontend.app import app
        app.run(debug=True)
    else:
        print("❌ Opción no válida")