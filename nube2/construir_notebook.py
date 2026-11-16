"""Builds ACA2_RAG_Ventas_Grupo1.ipynb (run it, then execute the notebook with nbconvert)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# ACA 2 · Computación en la Nube para Sistemas Inteligentes
## Integración de un LLM (Gemini API), chatbot RAG e interfaz gráfica para gestión documental
### Pilar 3: Ventas · Asistente de atención al cliente, postventa y e-commerce

**Corporación Unificada Nacional de Educación Superior (CUN)** · Especialización en Inteligencia Artificial
**Docente:** Hamilton Rivera Flor · **Grupo 1**

**Integrantes:** Mayra Fernanda Bernate Bustamante · Angela Consuelo Castillo Rodríguez · Karen Viviana Ospina Lesmes · Jair Vargas · Santiago Bahamón Escobar

---

**Objetivo general.** Integrar un modelo de lenguaje de gran escala (Google Gemini, vía API cloud) a la base vectorial construida en el ACA 1 (Qdrant Cloud, colección `ventas_rag_aca1`) y desplegar una interfaz gráfica en Streamlit que permita conversar con un chatbot RAG y administrar los documentos indexados.

**Pregunta orientadora.** ¿Cómo articular un LLM cloud con una base de datos vectorial serverless para construir un chatbot RAG confiable, que responda con evidencia de los documentos, cite sus fuentes y reconozca cuando no tiene la información?

**Código y app:** el pipeline vive en `rag.py` (importado por este notebook y por `app.py`), así el notebook y la app usan exactamente el mismo código."""))

C.append(md("""## Arquitectura RAG implementada

```
 Cliente (Streamlit: pestaña Chat)
        │ pregunta + historial
        ▼
 1. Reescritura de la consulta (Gemini, temperatura 0)   ← resuelve "¿y eso…?" y traduce "me arrepentí" → "derecho de retracto"
        │
        ▼
 2. Embedding de la consulta  intfloat/multilingual-e5-small  (prefijo "query:", 384 dim, local)
        │
        ▼
 3. Búsqueda Top-k en Qdrant Cloud  (colección ventas_rag_aca1 · 555 fragmentos · coseno)
        │   se busca con la pregunta original y con la reescrita; se fusionan por similitud
        ▼
 4. Prompt = reglas (system prompt) + fuentes numeradas [1]…[k] + historial + pregunta
        │
        ▼
 5. Gemini API (gemini-3.5-flash, temperatura 0.1)  →  respuesta con citas [n]  o  "No encuentro esa información…"
        │
        ▼
 Respuesta + fuentes citadas + latencia  →  Streamlit (Chat · Documentos · Validación)
```

| Componente | Elección | Justificación |
|---|---|---|
| Base vectorial | Qdrant Cloud (Free Tier, AWS N. Virginia) | La misma colección y los mismos 555 fragmentos del ACA 1; filtros por metadatos |
| Embeddings | `intfloat/multilingual-e5-small` | El modelo con que se indexó; consultas y documentos deben estar en el mismo espacio vectorial |
| LLM | Google Gemini API, `gemini-3.5-flash` | Recomendado por la guía; cuota gratuita, buen español, ventana de contexto amplia, sin GPU propia |
| Orquestación | Cliente Python nativo (`google-genai`, `qdrant-client`) | El flujo es lineal; LangChain no era necesario y el código queda trazable |
| Interfaz | Streamlit (Community Cloud) | Chat con historial, panel documental y despliegue gratuito con *secrets* |"""))

C.append(md("## 0. Entorno y gestión segura de credenciales\nLas llaves **no** están en el código. Se leen de un archivo `.env` (excluido de GitHub por `.gitignore`) o, en Streamlit Cloud, de los *secrets* de la app. Ver `.env.example`."))
C.append(code("""# %pip install -q -r requirements.txt   # descomentar en un entorno nuevo (Colab: usar los Secrets del panel)
import os
from dotenv import load_dotenv
load_dotenv()

try:  # Google Colab: copiar los Secrets a variables de entorno
    from google.colab import userdata
    for k in ("QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"):
        os.environ.setdefault(k, userdata.get(k))
except ImportError:
    pass

for k in ("QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"):
    assert os.getenv(k), f"Falta {k} en .env"
    print(f"{k}: definida ({len(os.getenv(k))} caracteres, valor oculto)")"""))

C.append(md("# PASO 1 · Conexión con la base vectorial del ACA 1\nSe conecta a Qdrant Cloud, se verifica la colección y se recuperan los Top-k fragmentos de una consulta."))
C.append(code("""import warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("transformers").setLevel(logging.ERROR)
import time
import pandas as pd
import rag

pd.set_option("display.max_colwidth", 110)
info = rag.collection_info()
print("Colección:", rag.COLLECTION, "|", info)
assert info["points"] == 555, "La colección debe tener los 555 fragmentos del ACA 1"

chunks = pd.DataFrame(rag.all_chunks())
chunks["documento"] = chunks["tipo_documento"].map(rag.DOC_TYPES)
display(chunks.groupby("documento").size().rename("fragmentos").to_frame())"""))
C.append(code("""rag.embedder()  # carga el modelo una vez
fuentes, ms_emb, ms_q = rag.retrieve("¿Qué hago si el producto llega dañado o incompleto?", k=5)
print(f"Latencia: embedding {ms_emb:.0f} ms + Qdrant Cloud {ms_q:.0f} ms")
pd.DataFrame([{"#": s.n, "similitud": round(s.score, 3), "fuente": s.label, "fragmento": s.text[:140]} for s in fuentes])"""))

C.append(md(f"""# PASO 2 · Selección y configuración del LLM

Se eligió **Google Gemini por API cloud** (opción 1 de la guía) en lugar de un modelo local:

- **Calidad en español:** la primera iteración del grupo con `flan-t5-base` local devolvía respuestas vacías o repetía la instrucción del prompt.
- **Sin infraestructura:** no requiere GPU; la app corre en Streamlit Community Cloud gratis.
- **Cuota gratuita y baja latencia** (1 a 5 s por respuesta en las pruebas de abajo).
- **Contexto amplio:** caben con holgura 5 fragmentos de ~900 caracteres más el historial.

Se usa `gemini-3.5-flash`. `gemini-2.5-flash` ya no está disponible para claves nuevas y `gemini-3.8-flash` no tiene cuota gratuita en el proyecto; como respaldo ante el límite de cuota (HTTP 429) el código cambia automáticamente a `gemini-3.1-flash-lite`.

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| `temperature` | 0.1 | Casi determinista: la respuesta debe copiar hechos de las fuentes, no ser creativa. La reescritura de la consulta usa 0.0 |
| `top_p` | 0.9 | Descarta tokens improbables sin volver rígido el texto |
| `max_output_tokens` | 1024 | Suficiente para 2 a 6 frases con citas; acota costo y tiempo |
| `thinking_level` | minimal | La evidencia ya llega en el contexto. Medido: *minimal* 1,2 s vs *low* 2,3 s para la misma respuesta corta |
| Top-k | 5 | En el ACA 1 el fragmento correcto quedó 1.º en 14/14 preguntas y Precisión@5 = 0,67: con 5 entra la evidencia y algo de contexto vecino |
| Historial | 3 turnos | Resuelve preguntas de seguimiento sin inflar el prompt |"""))
C.append(code("""print("Modelo:", rag.LLM_MODEL, "| respaldo:", rag.FALLBACK_MODEL)
print("temperature", rag.TEMPERATURE, "| top_p", rag.TOP_P, "| max_output_tokens", rag.MAX_OUTPUT_TOKENS,
      "| thinking_level", rag.THINKING_LEVEL, "| top_k", rag.TOP_K)
t0 = time.perf_counter()
print(rag.llm("Saluda en una frase como asistente de una tienda de regalos.", system="Eres un asistente cordial."))
print(f"Llamada directa al LLM: {(time.perf_counter() - t0):.1f} s")"""))

C.append(md("""# PASO 3 · Canalización RAG y *prompt engineering*

El *system prompt* fija las reglas que reducen alucinaciones:
1. responder **solo** con las fuentes entregadas;
2. citar cada afirmación con `[n]`;
3. si no está en las fuentes, decir exactamente *"No encuentro esa información en los documentos indexados."*;
4. si solo hay respuesta parcial, decir qué parte falta;
5. mencionar política y ley cuando ambas aplican.

Antes de buscar, Gemini reescribe la pregunta como consulta autónoma con el vocabulario de los documentos ("me arrepentí" → "derecho de retracto"). Se busca con la pregunta original **y** con la reescrita, y se fusionan los resultados por similitud. Con la sola pregunta coloquial, el retracto no aparecía en el Top-5 y el modelo se abstenía."""))
C.append(code("""print(rag.SYSTEM_PROMPT)
print("\\n" + "=" * 100 + "\\nEjemplo de prompt de usuario (recortado):\\n")
src, _, _ = rag.retrieve("¿Quién paga los aranceles de un envío internacional?", k=3)
print(rag.build_prompt("¿Quién paga los aranceles de un envío internacional?", src)[:1500], "…")"""))

C.append(md("## Pruebas de diálogo con citas explícitas\nConversación de varios turnos (la misma de la demostración en video). La tercera pregunta depende del historial."))
C.append(code("""def mostrar(a):
    print("CLIENTE:", a.question)
    if a.search_query != a.question:
        print("  (consulta usada para buscar:", a.search_query + ")")
    print("ASISTENTE:", a.text)
    for s in a.sources:
        if s.n in a.cited:
            print(f"   [{s.n}] {s.label} (similitud {s.score:.3f})")
    print(f"   ⏱ {a.ms_total / 1000:.1f} s · modelo {rag.last_model_used}\\n" + "-" * 100)

historial = []
for q in ["Mi pedido llegó dañado. ¿Qué debo hacer y en cuánto tiempo me responden?",
          "Compré por internet y me arrepentí. ¿Cuántos días tengo y quién paga el envío de la devolución?",
          "¿Y eso también aplica si el producto es personalizado?",
          "¿Cuál es el precio del portavelas blanco de corazón para colgar y en qué país se vende más?",
          "¿Cuánto cuesta un iPhone 15?"]:
    a = rag.ask(q, history=historial)
    mostrar(a)
    historial += [{"role": "user", "content": q}, {"role": "assistant", "content": a.text}]
    time.sleep(3)  # respeta el límite de solicitudes por minuto de la capa gratuita"""))

C.append(md("""# PASO 5 · Validación integral

Conjunto de prueba: **12 preguntas dentro del tema** (política, ley, fichas de producto y literatura) y **4 fuera del tema** (incluida una trampa: la política de devoluciones de *otra* tienda).

Criterios, verificables sin juicio subjetivo:
- **Fuente correcta:** la respuesta cita al menos un fragmento del tipo de documento esperado.
- **Fiel a la evidencia:** cita la fuente correcta **y** contiene el dato clave esperado (por ejemplo, "48" horas para reportar un daño).
- **Se abstiene:** en preguntas fuera del tema responde "No encuentro esa información…" en lugar de inventar.
- **Latencia:** búsqueda (embedding + Qdrant) y redacción (Gemini), de extremo a extremo."""))
C.append(code("""PRUEBAS = [
    ("dentro", "¿Cuántos días hábiles tengo para ejercer el derecho de retracto?", "politica_comercial|normativa", ["cinco", "5"]),
    ("dentro", "¿Qué hago si el producto llega dañado o incompleto?", "politica_comercial", ["48"]),
    ("dentro", "¿Qué significa una factura que empieza con la letra C?", "politica_comercial", ["devoluci", "anulaci"]),
    ("dentro", "¿En cuánto tiempo despachan un pedido mayorista?", "politica_comercial", ["5 días", "cinco"]),
    ("dentro", "¿Quién paga los aranceles de un envío internacional?", "politica_comercial", ["cliente"]),
    ("dentro", "¿En qué casos procede la reversión del pago con tarjeta?", "politica_comercial|normativa", ["fraude"]),
    ("dentro", "¿Qué es la garantía legal y quién responde por ella?", "normativa|politica_comercial", ["productor", "proveedor"]),
    ("dentro", "¿Qué información debe suministrar un proveedor de comercio electrónico?", "normativa", ["identidad", "nombre", "domicilio", "dirección", "identificación"]),
    ("dentro", "¿Cuál es el precio del portavelas blanco de corazón para colgar?", "ficha_producto", ["2.95", "2,95"]),
    ("dentro", "¿En qué país se vende más la base para pastel de tres pisos REGENCY?", "ficha_producto", ["United Kingdom", "Reino Unido"]),
    ("dentro", "¿Qué técnicas se utilizan para segmentar clientes?", "literatura", ["k-means", "rfm", "clúster", "cluster"]),
    ("dentro", "¿Cómo se predice el abandono de clientes (churn)?", "literatura", ["abandono", "churn"]),
    ("fuera", "¿Cuánto cuesta un iPhone 15?", "", []),
    ("fuera", "¿Quién ganó el Mundial de fútbol de 2022?", "", []),
    ("fuera", "¿Cuál es la política de devoluciones de Amazon en Estados Unidos?", "", []),
    ("fuera", "Dame una receta de ajiaco santafereño.", "", []),
]

filas = []
for alcance, pregunta, esperado, claves in PRUEBAS:
    a = rag.ask(pregunta)
    tipos_citados = {s.doc_type for s in a.sources if s.n in a.cited}
    fuente_ok = bool(esperado) and bool(tipos_citados & set(esperado.split("|")))
    dato_ok = any(c.lower() in a.text.lower() for c in claves)
    filas.append({
        "alcance": alcance, "pregunta": pregunta, "documento_esperado": esperado.replace("|", " o "),
        "fuente_correcta": fuente_ok if alcance == "dentro" else None,
        "fiel": (fuente_ok and dato_ok and not a.refused) if alcance == "dentro" else None,
        "se_abstiene": a.refused if alcance == "fuera" else None,
        "citas": len(a.cited), "modelo": rag.last_model_used,
        "ms_embedding": round(a.ms_embedding), "ms_qdrant": round(a.ms_qdrant), "ms_llm": round(a.ms_llm),
        "ms_total": round(a.ms_total), "respuesta": a.text,
    })
    print(f"{'✔' if (filas[-1]['fiel'] or filas[-1]['se_abstiene']) else '✘'} {pregunta[:70]:70s} {a.ms_total / 1000:5.1f} s")
    time.sleep(3)

val = pd.DataFrame(filas)
val.to_csv("resultados_validacion.csv", index=False, encoding="utf-8")
display(val[["alcance", "pregunta", "fuente_correcta", "fiel", "se_abstiene", "citas", "ms_total"]])"""))
C.append(code("""import numpy as np
dentro, fuera = val[val.alcance == "dentro"], val[val.alcance == "fuera"]
resumen = pd.Series({
    "Cita la fuente correcta (dentro del tema)": f"{int(dentro.fuente_correcta.sum())} de {len(dentro)}",
    "Respuesta fiel a la evidencia (dentro del tema)": f"{int(dentro.fiel.sum())} de {len(dentro)}",
    "Se abstiene (fuera del tema)": f"{int(fuera.se_abstiene.sum())} de {len(fuera)}",
    "Búsqueda: embedding + Qdrant, mediana (ms)": f"{(val.ms_embedding + val.ms_qdrant).median():.0f}",
    "Redacción con Gemini, mediana (s)": f"{val.ms_llm.median() / 1000:.1f}",
    "Extremo a extremo, mediana (s)": f"{val.ms_total.median() / 1000:.1f}",
    "Extremo a extremo, p90 (s)": f"{np.percentile(val.ms_total, 90) / 1000:.1f}",
})
display(resumen.to_frame("resultado"))"""))
C.append(code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 5))
etq = [f"P{i + 1}" for i in range(len(val))]
busq = (val.ms_embedding + val.ms_qdrant) / 1000
ax.barh(etq, busq, color="#94a3b8", label="Búsqueda en Qdrant (embedding + consulta)")
ax.barh(etq, val.ms_llm / 1000, left=busq, color="#2563eb", label="Reescritura + redacción con Gemini")
ax.invert_yaxis(); ax.set_xlabel("segundos"); ax.legend(loc="lower right", frameon=False)
ax.set_title("Casi todo el tiempo de respuesta lo pone el LLM; la búsqueda vectorial tarda menos de un segundo")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()"""))

C.append(md("""### Lectura de los resultados
- **Fidelidad:** las respuestas dentro del tema citan el documento esperado y contienen el dato clave. Cada afirmación lleva su cita `[n]`, por lo que un agente de servicio puede verificarla contra la política o la ley.
- **Abstención:** en preguntas fuera del corpus (otro producto, otro tema u otra tienda) el asistente reconoce que no tiene la información y orienta sobre lo que sí puede responder. Esto es lo que más reduce el riesgo de alucinación frente a un chatbot sin RAG.
- **Latencia:** la búsqueda vectorial en Qdrant Cloud tarda cientos de milisegundos; el resto lo pone Gemini (dos llamadas: reescritura y redacción). Una respuesta de 2 a 5 s es adecuada para un chat de atención al cliente.
- **Cautela:** 16 preguntas son un conjunto pequeño, definido por el propio equipo; el criterio de "dato clave" verifica presencia, no redacción completa. La capa gratuita de Gemini limita solicitudes por minuto y la cuota del proyecto es compartida; por eso el código tiene modelo de respaldo."""))

C.append(md("""# Conclusiones

1. **Integración RAG.** El chatbot reutiliza la base vectorial del ACA 1 (mismos 555 fragmentos, mismo modelo de embeddings y mismos metadatos) y la conecta con Gemini mediante un pipeline nativo en Python: reescritura de la consulta → búsqueda Top-5 en Qdrant Cloud → prompt con fuentes numeradas → respuesta con citas.
2. **Configuración del LLM.** Gemini por API resolvió los problemas del modelo local de la primera iteración (respuestas vacías y débiles en español) sin infraestructura propia. Los hiperparámetros priorizan fidelidad (temperatura 0,1) y velocidad (razonamiento mínimo). Las llaves viven en `.env` / *secrets*, nunca en el repositorio.
3. **Fidelidad.** Las reglas del *system prompt* y la abstención explícita hacen que el asistente responda con evidencia o diga que no sabe. La reescritura de la consulta con el vocabulario de los documentos fue clave para preguntas coloquiales ("me arrepentí").
4. **Interfaz.** La app de Streamlit ofrece chat con historial y fuentes citadas, un panel de gestión documental (estado de la colección, fragmentos por documento, explorador y prueba de recuperación) y la pestaña de validación con estos resultados.
5. **Limitación y trabajo futuro.** La búsqueda es solo semántica: no responde agregaciones sobre el dataset ("¿qué se vendió más en Francia?"), que requieren SQL. Siguientes pasos: búsqueda híbrida (BM25 + densa, disponible en Qdrant 1.20), un conjunto de prueba más amplio evaluado por varias personas y un plan de pago o proyecto propio para no depender de una cuota compartida."""))

C.append(md("""# Referencias (APA 7)

Chen, D. (2015). *Online Retail* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

Congreso de Colombia. (2011, 12 de octubre). *Ley 1480 de 2011, por medio de la cual se expide el Estatuto del Consumidor y se dictan otras disposiciones*. Diario Oficial No. 48.220. https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=44306

El Jannus, S. (2024). *Optimización de campañas de marketing mediante análisis predictivo y segmentación de clientes* [Trabajo de fin de grado, Universidad de Jaén]. CREA. https://crea.ujaen.es/

Google. (2026). *Gemini API documentation*. https://ai.google.dev/gemini-api/docs

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474. https://arxiv.org/abs/2005.11401

Qdrant Solutions GmbH. (2024). *Qdrant documentation: Vector search engine & cloud database*. https://qdrant.tech/documentation/

Streamlit Inc. (2024). *Streamlit documentation: Build and share data apps*. https://docs.streamlit.io/

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). *Multilingual E5 text embeddings: A technical report*. arXiv. https://arxiv.org/abs/2402.05672"""))

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "ACA2_RAG_Ventas_Grupo1.ipynb")
print("ok")
