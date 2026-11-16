"""Asistente RAG de Ventas · ACA 2 Computación en la Nube · Grupo 1 CUN.

Run:  streamlit run app.py
Credentials: .env locally or st.secrets on Streamlit Community Cloud
(QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY).
"""
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Asistente RAG de Ventas · Grupo 1", page_icon="🛍️", layout="wide")

# Streamlit Cloud secrets -> environment variables (rag.py reads os.environ)
try:
    for _k in ("QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets.toml locally
    pass

import rag  # noqa: E402  (after secrets are in the environment)

BASE = Path(__file__).resolve().parent
ACCENT = "#2563eb"
MUTED = "#94a3b8"

EXAMPLES = [
    "Mi pedido llegó dañado. ¿Qué debo hacer y en cuánto tiempo me responden?",
    "Compré por internet y me arrepentí. ¿Cuántos días tengo y quién paga el envío de la devolución?",
    "¿Y eso también aplica si el producto es personalizado?",
    "¿Cuál es el precio del portavelas blanco de corazón para colgar y en qué país se vende más?",
    "¿Cuánto cuesta un iPhone 15?",
]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el modelo de embeddings (e5-small)…")
def warm_up():
    rag.embedder()
    rag.qdrant()
    return True


@st.cache_data(ttl=600, show_spinner="Leyendo la colección en Qdrant Cloud…")
def load_chunks() -> pd.DataFrame:
    df = pd.DataFrame(rag.all_chunks())
    df["documento"] = df["tipo_documento"].map(rag.DOC_TYPES).fillna(df["fuente"])
    df["referencia"] = df.apply(lambda r: rag.reference(r.dropna().to_dict()), axis=1)
    df["caracteres"] = df["texto"].str.len()
    return df


@st.cache_data(ttl=600)
def load_info() -> dict:
    return rag.collection_info()


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_validation(_mtime_key: float) -> pd.DataFrame | None:
    p = BASE / "resultados_validacion.csv"
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Sidebar: configuration panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuración del asistente")
    k = st.slider("Fragmentos recuperados (Top-k)", 3, 8, rag.TOP_K,
                  help="Cuántos fragmentos de la base vectorial recibe el LLM como evidencia. "
                       "Con 5, en el ACA 1 el fragmento correcto quedó primero en 14 de 14 preguntas.")
    tipos = st.multiselect("Buscar solo en", options=list(rag.DOC_TYPES), format_func=rag.DOC_TYPES.get,
                           placeholder="Todos los documentos",
                           help="Filtro por metadato `tipo_documento` en Qdrant. Vacío = todo el corpus.")
    st.divider()
    st.markdown("**Modelo de lenguaje**")
    st.markdown(f"`{rag.LLM_MODEL}` vía Google Gemini API")
    st.markdown(
        f"""
- **Temperatura {rag.TEMPERATURE}**: respuestas estables que copian los hechos de las fuentes.
- **Top-p {rag.TOP_P}**: descarta palabras improbables.
- **Máx. {rag.MAX_OUTPUT_TOKENS} tokens**: respuesta corta; controla costo y tiempo.
- **Razonamiento mínimo**: la evidencia ya llega en el contexto; ahorra segundos.
- **Respaldo** `{rag.FALLBACK_MODEL}` si se agota la cuota gratuita.
""")
    st.markdown("**Recuperación**")
    st.markdown(f"Qdrant Cloud · colección `{rag.COLLECTION}`  \nEmbeddings `{rag.EMBED_MODEL.split('/')[1]}` (384 dim)")
    st.caption("Las llaves se leen de `.env` o de los *secrets* de Streamlit. Nunca están en el código.")

st.title("🛍️ Asistente de ventas con RAG")
st.caption("Responde dudas de garantías, devoluciones, envíos, derechos del consumidor y productos del catálogo "
           "**solo con los documentos indexados**, y cita la fuente de cada afirmación. "
           "Grupo 1 · Especialización en IA · CUN")

try:
    warm_up()
except Exception as exc:  # noqa: BLE001
    st.error(f"No se pudo conectar con la base vectorial o cargar el modelo: {exc}")
    st.stop()

tab_chat, tab_docs, tab_val = st.tabs(["💬 Chat", "📚 Documentos", "✅ Validación"])


# ---------------------------------------------------------------------------
# Tab 1: chat
# ---------------------------------------------------------------------------
def render_sources(ans: dict):
    cited = set(ans["cited"])
    label = f"Fuentes ({len(cited)} citadas de {len(ans['sources'])} recuperadas)" if not ans["refused"] \
        else f"Fragmentos revisados ({len(ans['sources'])}), ninguno responde la pregunta"
    with st.expander(label):
        for s in ans["sources"]:
            mark = "✅ citada" if s["n"] in cited else "· no citada"
            st.markdown(f"**[{s['n']}] {s['label']}** — similitud {s['score']:.3f} {mark}")
            st.caption(s["text"][:700] + ("…" if len(s["text"]) > 700 else ""))
        if ans["search_query"] != ans["question"]:
            st.caption(f"🔎 Búsqueda usada (pregunta reescrita con el historial): *{ans['search_query']}*")


def latency_line(ans: dict) -> str:
    return (f"⏱️ {ans['ms_total'] / 1000:.1f} s en total · búsqueda en Qdrant {ans['ms_embedding'] + ans['ms_qdrant']:.0f} ms "
            f"· redacción con Gemini {ans['ms_llm'] / 1000:.1f} s")


with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    clicked = None
    c1, c2 = st.columns([4, 1])
    with c1.expander("💡 Preguntas de ejemplo (clic para enviarla)", expanded=not st.session_state.messages):
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                clicked = ex
    if c2.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])
            if m["role"] == "assistant" and "ans" in m:
                st.caption(latency_line(m["ans"]))
                render_sources(m["ans"])

    prompt = st.chat_input("Escribe tu pregunta sobre garantías, devoluciones, envíos o productos…") or clicked
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.spinner("Buscando en los documentos y redactando…"):
            try:
                a = rag.ask(prompt, history=history, k=k, doc_types=tipos or None)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                st.stop()
        ans = {"question": a.question, "search_query": a.search_query, "cited": a.cited,
               "refused": a.refused, "ms_embedding": a.ms_embedding, "ms_qdrant": a.ms_qdrant,
               "ms_llm": a.ms_llm, "ms_total": a.ms_total,
               "sources": [{"n": s.n, "label": s.label, "score": s.score, "text": s.text} for s in a.sources]}
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": a.text, "ans": ans})
        st.rerun()  # redraw so the history stays above the input box


# ---------------------------------------------------------------------------
# Tab 2: document management
# ---------------------------------------------------------------------------
with tab_docs:
    info = load_info()
    df = load_chunks()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estado de la colección", "🟢 Activa" if info["status"] == "green" else info["status"],
              help=f"Colección `{rag.COLLECTION}` en Qdrant Cloud (AWS us-west-1, capa gratuita).")
    m2.metric("Documentos fuente", df["tipo_documento"].nunique(),
              help="Política de la tienda, Ley 1480 de 2011, fichas de producto de UCI Online Retail y TFG de marketing.")
    m3.metric("Fragmentos indexados", f"{info['points']:,}".replace(",", "."),
              help="Trozos de texto de ~900 caracteres, cada uno con su vector y sus metadatos.")
    m4.metric("Dimensión del vector", info["dimension"], help=f"multilingual-e5-small · distancia {info['distance']}")

    resumen = (df.groupby("documento")
               .agg(fragmentos=("id", "count"), tipo=("tipo_documento", "first"), fuente=("fuente", "first"),
                    caracteres_promedio=("caracteres", "mean"))
               .reset_index().sort_values("fragmentos", ascending=False))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Fragmentos por documento**")
        bars = alt.Chart(resumen).mark_bar(color=ACCENT, cornerRadiusEnd=3).encode(
            x=alt.X("fragmentos:Q", title=None, axis=None),
            y=alt.Y("documento:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            tooltip=["documento", "fragmentos", "fuente"])
        labels = bars.mark_text(align="left", dx=4, color=MUTED, fontWeight="bold").encode(text="fragmentos:Q")
        st.altair_chart((bars + labels).properties(height=200), use_container_width=True)
    with right:
        st.markdown("**Metadatos por documento**")
        st.dataframe(resumen[["documento", "tipo", "fragmentos", "fuente"]], hide_index=True,
                     use_container_width=True)

    st.divider()
    st.markdown("#### 🔎 Explorar fragmentos")
    f1, f2 = st.columns([2, 3])
    doc_sel = f1.selectbox("Documento", ["Todos"] + resumen["documento"].tolist())
    texto = f2.text_input("Contiene el texto", placeholder="p. ej. retracto, garantía, T-LIGHT…")
    vista = df if doc_sel == "Todos" else df[df["documento"] == doc_sel]
    if texto:
        vista = vista[vista["texto"].str.contains(texto, case=False, na=False, regex=False)]
    st.caption(f"{len(vista)} fragmentos")
    st.dataframe(vista[["documento", "referencia", "caracteres", "texto"]], hide_index=True,
                 use_container_width=True, height=280,
                 column_config={"texto": st.column_config.TextColumn("texto", width="large")})

    st.divider()
    st.markdown("#### 🧪 Probar la recuperación (sin LLM)")
    st.caption("Muestra qué fragmentos entrega Qdrant para una pregunta, con su similitud coseno. "
               "Es la evidencia que recibiría Gemini.")
    q = st.text_input("Pregunta de prueba", value="¿Qué hago si el producto llega dañado?")
    if q:
        srcs, ms_e, ms_q = rag.retrieve(q, k, tipos or None)
        st.caption(f"⏱️ embedding {ms_e:.0f} ms + Qdrant Cloud {ms_q:.0f} ms")
        st.dataframe(pd.DataFrame([{"#": s.n, "similitud": round(s.score, 3), "documento": rag.DOC_TYPES.get(s.doc_type),
                                    "referencia": s.reference, "fragmento": s.text[:300]} for s in srcs]),
                     hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: validation results (produced by the notebook)
# ---------------------------------------------------------------------------
with tab_val:
    val = load_validation(_mtime(BASE / "resultados_validacion.csv"))
    if val is None:
        st.info("Ejecute el notebook `ACA2_RAG_Ventas_Grupo1.ipynb` para generar `resultados_validacion.csv`.")
    else:
        dentro = val[val["alcance"] == "dentro"]
        fuera = val[val["alcance"] == "fuera"]
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Cita la fuente correcta", f"{int(dentro['fuente_correcta'].sum())} de {len(dentro)}",
                  help="Preguntas del tema en las que la respuesta cita al menos un fragmento del documento esperado.")
        v2.metric("Respuestas fieles a la evidencia", f"{int(dentro['fiel'].sum())} de {len(dentro)}",
                  help="La respuesta contiene el dato clave esperado y cada afirmación lleva cita [n].")
        v3.metric("Se abstiene fuera del tema", f"{int(fuera['se_abstiene'].sum())} de {len(fuera)}",
                  help="Preguntas que los documentos no cubren: el asistente dice que no tiene la información en vez de inventar.")
        v4.metric("Tiempo típico de respuesta", f"{val['ms_total'].median() / 1000:.1f} s",
                  help="Mediana de extremo a extremo: búsqueda en Qdrant + redacción con Gemini.")

        ok_total = int(dentro["fiel"].sum()) + int(fuera["se_abstiene"].sum())
        st.info(f"**Lectura ejecutiva.** En {ok_total} de {len(val)} preguntas de prueba el asistente se comportó como "
                "debe: respondió con el dato del documento correcto y lo citó, o reconoció que no tenía la información. "
                "Para un área de servicio al cliente esto significa respuestas que se pueden auditar contra la política "
                "y la ley, sin respuestas inventadas.")

        lat = val.assign(busqueda_ms=val["ms_embedding"] + val["ms_qdrant"])
        l1, l2 = st.columns(2)
        l1.metric("Búsqueda en Qdrant (mediana)", f"{lat['busqueda_ms'].median():.0f} ms")
        l2.metric("Redacción con Gemini (mediana)", f"{lat['ms_llm'].median() / 1000:.1f} s")

        with st.expander("Detalle por pregunta"):
            cols = ["alcance", "pregunta", "documento_esperado", "fuente_correcta", "fiel", "se_abstiene",
                    "citas", "ms_total", "respuesta"]
            st.dataframe(val[[c for c in cols if c in val.columns]], hide_index=True, use_container_width=True,
                         column_config={"ms_total": st.column_config.NumberColumn("ms total", format="%d"),
                                        "respuesta": st.column_config.TextColumn("respuesta", width="large")})
