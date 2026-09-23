import spacy
from bs4 import BeautifulSoup
import numpy as np
from loguru import logger

# Cargamos el modelo => ejecutar en terminal: python -m spacy download es_core_news_md
try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    logger.error(
        "Modelo de Spacy no encontrado. Ejecuta: python -m spacy download "
        "es_core_news_md"
    )
    raise


def limpiar_html(ruta):
    """
    Extrae el texto puro del HTML.
    """

    with open(ruta, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    # Extraemos todo el texto, eliminando etiquetas
    return soup.get_text(separator=" ", strip=True)


def crear_chunks(texto, chunk_size=150, overlap=40):
    """
    Divide el texto en chunks de 'chunk_size' palabras, solapando 'overlap' palabras
    entre chunks contiguos.
    """

    palabras = texto.split()
    chunks = []
    paso = chunk_size - overlap

    for i in range(0, len(palabras), paso):
        chunk = " ".join(palabras[i : i + chunk_size])
        chunks.append(chunk)
        # Si ya hemos cogido las últimas palabras, paramos
        if i + chunk_size >= len(palabras):
            break

    return chunks


def procesar_chunk(texto):
    """
    Lematiza, elimina stopwords y calcula el vector promedio. Retorna un
    diccionario preparado para ser indexado.
    """

    doc = nlp(texto.lower())

    lemas = []
    vectores = []

    for t in doc:
        if t.is_stop or t.is_punct or t.is_space:
            continue
        lemas.append(t.lemma_)
        if t.has_vector:
            vectores.append(t.vector)

    # Calculamos el vector promedio si hay vectores, si no, None
    vector = (
        np.mean(vectores, axis=0) if vectores else np.zeros(nlp.vocab.vectors_length)
    )

    return {"texto_original": texto, "lemas": lemas, "vector": vector}


def construir_corpus(ruta):
    """
    Pipeline completo de preprocesado.
    """

    logger.info("Preprocesando corpus...")
    texto = limpiar_html(ruta)
    chunks = crear_chunks(texto)

    corpus = []
    for c in chunks:
        data = procesar_chunk(c)
        # Solo guardamos chunks que tengan contenido válido
        if data["lemas"]:
            corpus.append(data)

    logger.success(f"Corpus listo con {len(corpus)} chunks")
    return corpus
