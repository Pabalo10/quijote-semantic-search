# P4: El que lee mucho y anda mucho, ve mucho y sabe mucho 📖🤺
**Buscador Clásico, Semántico y RAG para El Quijote**

## 👥 Equipo de Desarrollo
* **Pablo Alonso Romero**
* **Rodrigo Jesús-Portanet Martínez** 

---

## 📝 Descripción del Proyecto
Este proyecto implementa un sistema de Recuperación de Información (Information Retrieval) y Procesamiento de Lenguaje Natural (NLP) sobre el corpus literario de *Don Quijote de la Mancha*. A través de una Interfaz de Usuario de Terminal (TUI) interactiva, el usuario puede realizar consultas sobre el texto utilizando diferentes paradigmas de búsqueda, desde la coincidencia léxica hasta la comprensión semántica profunda y la generación de respuestas mediante Inteligencia Artificial (RAG).

## ✨ Características y Modos de Búsqueda

El sistema ofrece tres modos de operación principales:

1. **Búsqueda Clásica (TF-IDF):** Encuentra pasajes relevantes mediante similitud léxica. El motor preprocesa la consulta y el corpus eliminando palabras vacías (*stopwords*) y signos de puntuación, y lematiza los términos. Utiliza la métrica TF-IDF (Term Frequency - Inverse Document Frequency) para puntuar y ordenar (rankear) los resultados por relevancia.

2. **Búsqueda Semántica (Embeddings):**
   Busca por el "fondo" o significado de la consulta. Utiliza representaciones vectoriales densas (*word embeddings* a través del modelo `es_core_news_md` de spaCy). Se calcula el vector promedio de la consulta y se compara con los vectores de los *chunks* del texto mediante similitud coseno, devolviendo los pasajes semánticamente más cercanos.

3. **RAG (Retrieval-Augmented Generation):**
   Combina la potencia de la búsqueda semántica con un Modelo de Lenguaje Grande (LLM) local instanciado mediante Ollama. El sistema recupera los pasajes más relevantes y los inyecta como contexto en un *prompt* estricto, obligando al LLM a generar una respuesta en lenguaje natural basada **únicamente** en el texto recuperado y referenciando explícitamente sus fuentes.

### 🚀 Optimización y Caché
Para garantizar una experiencia de usuario fluida, el sistema preprocesa el texto fuente dividiéndolo en *chunks* (con solapamiento para no perder contexto) y precalcula los lemas y embeddings. Este corpus estructurado se guarda en caché (`cache/corpus.pkl`). 
El sistema es completamente autónomo y capaz de **regenerar estas estructuras bajo demanda** mediante el flag `--rebuild`, asegurando la reproducibilidad total sin intervención manual en el sistema de archivos.

---

## 🧹 Preprocesado de Datos

Para asegurar que el motor de búsqueda y el LLM ofrezcan resultados precisos y estrictamente relacionados con la obra, el tratamiento del texto original se ha dividido en dos fases:

### 1. Limpieza Manual
El texto original se obtuvo a partir de un archivo comprimido (`.zip`). Tras su extracción, se realizó una criba manual directamente sobre el archivo HTML (`data/2000-h.htm`). En este paso **se eliminaron todas aquellas partes que no pertenecían a la narración principal**. De esta manera, garantizamos que el motor no indexe ruido ni información externa a la novela en sí.

### 2. Preprocesado Automático (Pipeline en código)
Una vez preparado el documento base, el sistema ejecuta un preprocesado automatizado mediante código Python antes de construir el índice de búsqueda:
* **Extracción de texto puro:** Se utiliza la librería `BeautifulSoup` para parsear el documento, eliminar todas las etiquetas HTML y extraer el contenido limpio.
* **Chunking con solapamiento:** El texto íntegro se divide en fragmentos (*chunks*) de 150 palabras, manteniendo un solapamiento (*overlap*) de 40 palabras entre bloques contiguos. Esta técnica es vital para evitar que el sentido de una frase se pierda si ocurre un salto de bloque.
* **Procesamiento NLP:** Mediante `spaCy`, cada *chunk* es tokenizado y filtrado para descartar *stopwords*, espacios y signos de puntuación. Finalmente, se extraen los lemas de las palabras restantes (necesarios para la Búsqueda Clásica) y se calcula el vector promedio (*embedding*) de todo el fragmento (necesario para la Búsqueda Semántica).

---

## 📂 Estructura del Proyecto

```text
p4/
├── pyproject.toml         # Configuración del proyecto, dependencias y scripts
├── uv.lock                # Archivo de bloqueo de dependencias de uv
├── README.md              # Documentación del proyecto
├── cache/                 # Carpeta autogenerada para persistencia de datos
│   └── corpus.pkl         # Corpus preprocesado, lematizado y vectorizado
├── data/                  # Carpeta de datos fuente
│   └── 2000-h.htm         # Texto original íntegro de El Quijote en formato HTML
└── src/                   # Código fuente de la aplicación
    ├── app.py             # Interfaz TUI desarrollada con Textual
    ├── cli.py             # Interfaz de Línea de Comandos (Typer) para opciones de ejecución
    ├── motor_busqueda.py  # Lógica algorítmica: TF-IDF y Similitud Coseno
    ├── preprocesado.py    # Extracción HTML, chunking, lematización y word embeddings
    └── rag.py             # Integración con LLM local vía Ollama
```

---

## ⚙️ Requisitos e Instalación

El proyecto utiliza `uv` como gestor de paquetes y dependencias. El código requiere **Python >= 3.12**.

**1. Instalar dependencias:**
Ejecuta el siguiente comando en la raíz del proyecto para sincronizar el entorno virtual:
```bash
uv sync
```

**2. Requisitos de Modelos (NLP e IA):**

* **spaCy:** El modelo de embeddings en español se descarga automáticamente a través de la configuración del `pyproject.toml`. En caso de fallo manual, ejecutar:
  ```bash
  uv run python -m spacy download es_core_news_md
  ```

* **Ollama:** Para que el modo RAG funcione, es imprescindible tener Ollama instalado y corriendo en el sistema con el modelo `llama3` (o el configurado) descargado:
  ```bash
  ollama run llama3
  ```

## 💻 Ejecución y Uso

La aplicación se expone a través de un único comando principal instalado en el entorno.

**1. Iniciar la Interfaz TUI:**
Ejecuta el programa normalmente. Si es la primera vez (o no existe la carpeta `cache`), el sistema preprocesará el corpus automáticamente (puede tardar un par de minutos).
```bash
uv run fdi-pln-2603-p4
```

**2. Forzar la Reconstrucción del Corpus:**
Si el texto fuente cambia, o se requiere recalcular los vectores y el IDF por motivos de desarrollo, se puede forzar la eliminación de la caché y la regeneración completa utilizando el flag de reconstrucción:
```bash
uv run fdi-pln-2603-p4 --rebuild
```
*(También se puede utilizar la versión corta `-r`)*

---

## 🛠️ Calidad del Código (Linting & Formatting)

El código cumple con los más altos estándares de estilo. Para verificar o aplicar el formato requerido por la práctica, utiliza:
```bash
uv format
uv format --check
```

---
