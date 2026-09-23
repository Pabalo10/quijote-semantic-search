import ollama


def generar_respuesta_rag(consulta, resultados, modelo="llama3"):
    """
    Genera una respuesta utilizando un LLM local vía Ollama. Usa los resultados
    de la búsqueda como contexto y fuerza el uso de referencias.
    """

    contexto = ""
    for i, r in enumerate(resultados[:5]):
        contexto += f"[Ref {i + 1}] {r['texto']}\n\n"

    prompt = f"""
Eres un asistente experto en literatura española.

Responde usando SOLO el contexto proporcionado.
Si la respuesta no aparece en el contexto, di: "No se encuentra en el texto".

Pregunta: {consulta}

Contexto:
{contexto}

Respuesta:
"""

    try:
        response = ollama.chat(
            model=modelo, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    except Exception as e:
        return f"Error con Ollama: {e}"
