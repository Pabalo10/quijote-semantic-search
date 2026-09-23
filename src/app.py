import os
import pickle
import spacy
import numpy as np

from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Input,
    RadioSet,
    RadioButton,
    Button,
    Static,
    Markdown,
)
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.binding import Binding

from src.preprocesado import construir_corpus
from src.motor_busqueda import MotorBusqueda
from src.rag import generar_respuesta_rag

# Archivo donde guardaremos el corpus para no esperar 3 minutos cada vez
CACHE = "cache/corpus.pkl"


class QuijoteApp(App):
    BINDINGS = [
        Binding("ctrl+q", "quit", "Salir", show=True),
    ]

    def __init__(self, rebuild: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuild = rebuild

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        # yield Input(placeholder="Consulta...")
        # yield Button("Buscar")
        # yield Markdown("Cargando...")

        # Panel superior de controles
        with Vertical(id="panel-control"):
            yield Static("Buscador Inteligente: El Quijote", classes="titulo")
            yield Input(
                placeholder="Introduce tu búsqueda (ej. molinos de viento gigantes)",
                id="input_consulta",
            )

            with Horizontal():
                with RadioSet(id="selector_modo"):
                    yield RadioButton(
                        "Búsqueda Clásica (TF-IDF)", value=True, id="modo_clasica"
                    )
                    yield RadioButton(
                        "Búsqueda Semántica (Embeddings)", id="modo_semantica"
                    )
                    yield RadioButton("RAG (Búsqueda + IA)", id="modo_rag")
                yield Button("Buscar", id="btn_buscar", variant="primary")

        # Panel inferior para mostrar resultados
        with VerticalScroll(id="panel-resultados"):
            yield Markdown(
                "Esperando a que se cargue el sistema...", id="pantalla_resultados"
            )

        yield Footer()

    def on_mount(self):
        """Se ejecuta al abrir la aplicación."""
        self.nlp = spacy.load("es_core_news_md")

        if os.path.exists(CACHE) and not self.rebuild:
            with open(CACHE, "rb") as f:
                corpus = pickle.load(f)
        else:
            corpus = construir_corpus("data/2000-h.htm")
            os.makedirs("cache", exist_ok=True)
            with open(CACHE, "wb") as f:
                pickle.dump(corpus, f)

        self.motor = MotorBusqueda(corpus)

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id != "btn_buscar":
            return

        consulta = self.query_one("#input_consulta", Input).value.strip()

        if not consulta:
            self.query_one("#pantalla_resultados", Markdown).update(
                "⚠️ Introduce una consulta."
            )
            return

        # Procesar consulta
        doc = self.nlp(consulta)

        lemas = [t.lemma_ for t in doc if not t.is_stop]
        vectores = [t.vector for t in doc if t.has_vector and not t.is_stop]
        vq = (
            np.mean(vectores, axis=0)
            if vectores
            else np.zeros(self.nlp.vocab.vectors_length)
        )

        # Saber qué modo está seleccionado
        modo = self.query_one("#selector_modo", RadioSet).pressed_button.id

        resultados = []
        texto = ""

        # BÚSQUEDA CLÁSICA
        if modo == "modo_clasica":
            resultados = self.motor.busqueda_clasica(lemas)
            texto += "## Búsqueda Clásica (TF-IDF)\n"

        # BÚSQUEDA SEMÁNTICA
        elif modo == "modo_semantica":
            resultados = self.motor.busqueda_semantica(vq)
            texto += "## Búsqueda Semántica\n"

        # RAG
        elif modo == "modo_rag":
            resultados = self.motor.busqueda_semantica(vq)
            texto += "## RAG (Generación con IA)\n"

            respuesta = generar_respuesta_rag(consulta, resultados)
            texto += f"\n### Respuesta generada:\n{respuesta}\n\n---\n"

        # Mostrar resultados
        if not resultados:
            texto += "\n No se encontraron resultados."
        else:
            for i, r in enumerate(resultados):
                texto += f"\n\n### Resultado {i + 1} (score: {r['score']:.3f})\n"
                texto += f"{r['texto'][:300]}..."

        self.query_one("#pantalla_resultados", Markdown).update(texto)


def main(rebuild: bool = False):
    QuijoteApp(rebuild=rebuild).run()


if __name__ == "__main__":
    main()
