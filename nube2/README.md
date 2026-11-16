# ACA 2 · Computación en la Nube · Chatbot RAG de Ventas (Grupo 1, CUN)

Asistente de atención al cliente que responde dudas de **garantías, devoluciones, envíos, derechos del consumidor y productos del catálogo** usando solo los documentos indexados en el ACA 1, y cita la fuente de cada afirmación.

- **App publicada:** APP_URL
- **Video de sustentación:** VIDEO_URL
- Docente: Dr. Hamilton Rivera Flor · Integrantes: Mayra Fernanda Bernate Bustamante · Angela Consuelo Castillo Rodríguez · Karen Viviana Ospina Lesmes · Jair Vargas · Santiago Bahamón Escobar

## Arquitectura

```
Pregunta (Streamlit) ─► Gemini reescribe la consulta ─► e5-small ("query:") ─► Qdrant Cloud Top-5
      ▲                                                                         │
      └──── respuesta con citas [n] + fuentes + latencia ◄── Gemini (temp 0.1) ◄─┘ prompt con fuentes numeradas
```

| Componente | Elección |
|---|---|
| Base vectorial | Qdrant Cloud, colección `ventas_rag_aca1` (555 fragmentos del ACA 1) |
| Embeddings | `intfloat/multilingual-e5-small` (384 dim, local) |
| LLM | Google Gemini API `gemini-3.5-flash` (respaldo `gemini-3.1-flash-lite` si se agota la cuota) |
| Hiperparámetros | temperatura 0.1 · top-p 0.9 · máx. 1024 tokens · razonamiento *minimal* · Top-k 5 · historial 3 turnos |
| Interfaz | Streamlit: **Chat** (historial + fuentes citadas), **Documentos** (estado de la colección, fragmentos por documento, explorador, prueba de recuperación) y **Validación** |

## Archivos

| Archivo | Qué hace |
|---|---|
| `app.py` | Interfaz Streamlit |
| `rag.py` | Pipeline RAG: conexión a Qdrant, embeddings, reescritura, prompt, llamada a Gemini, citas y latencia |
| `ACA2_RAG_Ventas_Grupo1.ipynb` | Notebook reproducible: conexión, configuración del LLM, pipeline, diálogos con citas y validación (16 preguntas) |
| `resultados_validacion.csv` | Resultados de la validación (los muestra la pestaña Validación) |
| `indexar.py` | Reconstruye la colección del ACA 1 desde `corpus/` si el clúster se perdió |
| `construir_notebook.py` | Genera el notebook |

## Cómo ejecutar

1. Copiar `.env.example` a `.env` y completar `QDRANT_URL`, `QDRANT_API_KEY` y `GEMINI_API_KEY` (Google AI Studio). **Nunca subir `.env`**; está en `.gitignore`.
2. Instalar dependencias (Python 3.12 o 3.13):
   ```
   pip install -r requirements.txt
   ```
3. Si la colección no existe en su clúster: `pip install -r requirements-notebook.txt` y `python indexar.py`.
4. Ejecutar la app:
   ```
   streamlit run app.py
   ```
5. Notebook: `jupyter nbconvert --to notebook --execute ACA2_RAG_Ventas_Grupo1.ipynb` (o abrirlo en Jupyter/Colab; en Colab usar los *Secrets*).

En **Streamlit Community Cloud** las mismas tres variables se cargan en *Settings → Secrets*:

```toml
QDRANT_URL = "https://..."
QDRANT_API_KEY = "..."
GEMINI_API_KEY = "..."
```

## Referencias

Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS, 33*, 9459–9474. https://arxiv.org/abs/2005.11401 · Qdrant Solutions GmbH. (2024). *Qdrant documentation*. https://qdrant.tech/documentation/ · Google. (2026). *Gemini API documentation*. https://ai.google.dev/gemini-api/docs · Streamlit Inc. (2024). *Streamlit documentation*. https://docs.streamlit.io/ · Wang, L., et al. (2024). *Multilingual E5 text embeddings*. https://arxiv.org/abs/2402.05672
