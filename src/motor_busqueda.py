import math
import numpy as np
from collections import Counter
from loguru import logger


class MotorBusqueda:
    def __init__(self, corpus):
        """
        Inicializa el motor con el corpus preprocesado. corpus: Lista de diccionarios
        devuelta por procesar_chunk()
        """

        self.corpus = corpus
        self.N = len(corpus)
        self.idf = {}

        logger.info("Inicializando Motor de Búsqueda...")
        self._calcular_idf()

    def _calcular_idf(self):
        """
        Calcula el Inverse Document Frequency para todos los lemas del corpus.
        """

        df = Counter()

        # Contamos en cuántos documentos (chunks) aparece cada lema
        for doc in self.corpus:
            # Usamos set() para no contar un término varias veces en el mismo chunk
            for lema in set(doc["lemas"]):
                df[lema] += 1

        # Calculamos el IDF
        for lema, freq in df.items():
            # Sumamos 1 en el denominador para evitar divisiones por cero por seguridad
            self.idf[lema] = math.log(self.N / (freq + 1))

    def busqueda_clasica(self, lemas_consulta, top_k=5):
        """
        Realiza una búsqueda basada en TF-IDF. lemas_consulta: Lista de lemas extraídos
        de la consulta del usuario.
        """

        resultados = []

        for i, doc in enumerate(self.corpus):
            score = 0
            contador = Counter(doc["lemas"])
            total = len(doc["lemas"])

            # Calculamos TF-IDF para cada término de la consulta
            for lema in lemas_consulta:
                # Term Frequency (TF): Veces que aparece el término / Total de términos en el documento
                if lema in contador:
                    tf = contador[lema] / total
                    # Multiplicamos por el IDF precalculado
                    score += tf * self.idf.get(lema, 0)

            if score > 0:
                resultados.append(
                    {"chunk_id": i, "texto": doc["texto_original"], "score": score}
                )

        # Ordenamos de mayor a menor score (TF-IDF más alto = más relevante)
        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:top_k]

    def busqueda_semantica(self, vector_q, top_k=5):
        """
        Realiza una búsqueda basada en embeddings y distancia coseno.
        vector_consulta: Numpy array con el embedding promedio de la consulta.
        """

        logger.info("Realizando búsqueda semántica...")
        resultados = []

        for i, doc in enumerate(self.corpus):
            v = doc["vector"]
            if np.all(vector_q == 0):
                logger.warning(
                    "El vector de la consulta es nulo. No se pueden buscar similitudes."
                )
                continue

            # Calculamos la Distancia Coseno entre dos vectores
            sim = np.dot(vector_q, v) / (
                np.linalg.norm(vector_q) * np.linalg.norm(v) + 1e-9
            )

            resultados.append(
                {"chunk_id": i, "texto": doc["texto_original"], "score": sim}
            )

        # Ordenamos de menor a mayor distancia (los más cercanos primero)
        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:top_k]
