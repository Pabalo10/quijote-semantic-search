# prueba.py
import spacy
from preprocesado import construir_corpus
from motor_busqueda import MotorBusqueda

# Cargar el mismo modelo que en preprocesado
nlp = spacy.load("es_core_news_md")


def vectorizar_consulta(texto):
    doc = nlp(texto.lower())
    vectores = [
        t.vector for t in doc if t.has_vector and not t.is_punct and not t.is_stop
    ]
    import numpy as np

    return np.mean(vectores, axis=0) if vectores else np.zeros(nlp.vocab.vectors_length)


def lematizar_consulta(texto):
    doc = nlp(texto.lower())
    return [t.lemma_ for t in doc if not t.is_punct and not t.is_stop]


# 1. Construir corpus (Asegúrate de tener un html corto o el Quijote para probar)
corpus = construir_corpus("2000-h.htm")

# 2. Inicializar motor
motor = MotorBusqueda(corpus)

# 3. Probar Búsqueda Clásica
consulta = "molinos de viento"
lemas = lematizar_consulta(consulta)
print("\n--- Búsqueda Clásica ---")
for res in motor.busqueda_clasica(lemas, top_k=2):
    print(f"Score: {res['score']:.4f} | Texto: {res['texto'][:100]}...")

# 4. Probar Búsqueda Semántica
vector = vectorizar_consulta(consulta)
print("\n--- Búsqueda Semántica ---")
for res in motor.busqueda_semantica(vector, top_k=2):
    print(f"Distancia: {res['score']:.4f} | Texto: {res['texto'][:100]}...")
