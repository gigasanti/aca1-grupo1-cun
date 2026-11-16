"""RAG pipeline for the Ventas assistant (ACA 2 · Computación en la Nube · Grupo 1 CUN).

Flow: question -> (optional) rewrite with chat history -> e5-small embedding ("query:" prefix)
-> Top-k search in Qdrant Cloud (collection built in ACA 1) -> prompt with numbered sources
-> Gemini -> answer with [n] citations + sources + latency.

Credentials are read from environment variables (.env locally, st.secrets on Streamlit Cloud):
QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY. Optional: GEMINI_MODEL.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (justified in README and in the notebook)
# ---------------------------------------------------------------------------
COLLECTION = "ventas_rag_aca1"           # vector base delivered in ACA 1
EMBED_MODEL = "intfloat/multilingual-e5-small"  # same model used to index (384 dim)
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = "gemini-3.1-flash-lite"  # used only if the main model hits the free-tier quota (HTTP 429)

TOP_K = 5             # ACA 1: Hit@1 14/14 and Precision@5 0.67 -> 5 chunks bring the right one plus useful neighbours
TEMPERATURE = 0.1     # near-deterministic: the answer must copy facts from the sources, not be creative
TOP_P = 0.9           # trims unlikely tokens without making the output rigid
MAX_OUTPUT_TOKENS = 1024  # enough for a 3-6 sentence answer with citations; caps cost and latency
THINKING_LEVEL = "minimal"  # internal reasoning almost off: the facts already come in the context;
                            # measured: minimal 1.2 s vs low 2.3 s for the same short answer
HISTORY_TURNS = 3     # last user/assistant pairs used to resolve follow-up questions

NOT_FOUND = "No encuentro esa información en los documentos indexados."

DOC_TYPES = {
    "politica_comercial": "Política de devoluciones, garantías y SLA",
    "normativa": "Ley 1480 de 2011 (Estatuto del Consumidor)",
    "ficha_producto": "Fichas de producto (UCI Online Retail)",
    "literatura": "TFG marketing y segmentación (U. de Jaén)",
}

SYSTEM_PROMPT = f"""Eres el asistente de atención al cliente y soporte comercial de una tienda en línea de artículos de regalo y decoración.
Respondes en español, con un tono claro, cordial y profesional.

Reglas obligatorias:
1. Responde ÚNICAMENTE con la información de las FUENTES que se te entregan. No uses conocimiento propio ni inventes datos, cifras, plazos o artículos de ley.
2. Cada afirmación debe llevar la cita de la fuente entre corchetes, por ejemplo [1] o [2][3], justo después de la frase que respalda.
3. Si las fuentes no contienen la respuesta, responde exactamente: "{NOT_FOUND}" y, si aplica, sugiere en una frase qué tipo de pregunta sí puedes responder (garantías, devoluciones, envíos, derechos del consumidor, productos del catálogo o segmentación de clientes). No añadas citas en ese caso.
4. Si las fuentes solo cubren una parte de la pregunta, responde esa parte con citas y aclara qué parte no está en los documentos.
5. Si la política de la tienda y la Ley 1480 dicen cosas complementarias, menciona ambas con su cita.
6. Responde como en un chat: una o dos frases cortas y directas, con el dato que el cliente necesita. Usa una lista corta solo si pide varios elementos. Las cifras de dinero de las fichas de producto están en libras esterlinas (£)."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Source:
    n: int
    score: float
    doc_type: str
    reference: str
    source: str
    text: str
    payload: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{DOC_TYPES.get(self.doc_type, self.source)} · {self.reference}"


@dataclass
class Answer:
    question: str
    search_query: str
    text: str
    sources: list[Source]
    cited: list[int]
    ms_embedding: float
    ms_qdrant: float
    ms_llm: float
    refused: bool

    @property
    def ms_total(self) -> float:
        return self.ms_embedding + self.ms_qdrant + self.ms_llm


# ---------------------------------------------------------------------------
# Clients (cached: loading e5 takes a few seconds, the clients are reused)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def qdrant():
    from qdrant_client import QdrantClient
    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url or not key:
        raise RuntimeError("Defina QDRANT_URL y QDRANT_API_KEY en .env o en los secrets de Streamlit")
    return QdrantClient(url=url, api_key=key, timeout=30)


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    model.max_seq_length = 512
    return model


@lru_cache(maxsize=1)
def gemini():
    from google import genai
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Defina GEMINI_API_KEY en .env o en los secrets de Streamlit")
    return genai.Client(api_key=key)


def generation_config(system: str, model: str, temperature: float = TEMPERATURE,
                      max_tokens: int = MAX_OUTPUT_TOKENS):
    from google.genai import types
    kwargs = dict(system_instruction=system, temperature=temperature, top_p=TOP_P,
                  max_output_tokens=max_tokens, http_options=types.HttpOptions(timeout=30000))
    if "lite" not in model:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    return types.GenerateContentConfig(**kwargs)


last_model_used = LLM_MODEL


def llm(prompt: str, system: str = SYSTEM_PROMPT, temperature: float = TEMPERATURE,
        max_tokens: int = MAX_OUTPUT_TOKENS, retries: int = 2) -> str:
    """Gemini call with retry; on quota errors (429) it falls back to the lite model."""
    global last_model_used
    last = None
    for round_ in range(2):  # second round only if both models hit the per-minute quota
        for model in (LLM_MODEL, FALLBACK_MODEL):
            for attempt in range(retries):
                try:
                    resp = gemini().models.generate_content(
                        model=model, contents=prompt,
                        config=generation_config(system, model, temperature, max_tokens))
                    last_model_used = model
                    return (resp.text or "").strip()
                except Exception as exc:  # noqa: BLE001 - surface the last error after retries
                    last = exc
                    if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                        break  # quota: go straight to the fallback model
                    time.sleep(1.5 * (attempt + 1))
        if round_ == 0:
            time.sleep(12)  # free tier asks to retry in ~10 s
    raise RuntimeError(f"Error al llamar a Gemini: {last}")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def reference(payload: dict) -> str:
    if payload.get("stock_code"):
        return f"producto {payload['stock_code']} ({payload.get('descripcion', '').strip()})"
    if payload.get("seccion"):
        return f"sección {payload['seccion']}"
    if payload.get("pagina") is not None:
        return f"pág. {int(payload['pagina'])}"
    return "—"


def retrieve(query: str, k: int = TOP_K, doc_types: list[str] | None = None):
    """Top-k chunks from Qdrant Cloud. Returns (sources, ms_embedding, ms_qdrant)."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    t0 = time.perf_counter()
    vector = embedder().encode(f"query: {query}", normalize_embeddings=True).tolist()
    t1 = time.perf_counter()
    flt = None
    if doc_types:
        flt = Filter(must=[FieldCondition(key="tipo_documento", match=MatchAny(any=list(doc_types)))])
    points = qdrant().query_points(COLLECTION, query=vector, limit=k, query_filter=flt,
                                   with_payload=True).points
    t2 = time.perf_counter()
    sources = [Source(n=i, score=float(p.score), doc_type=p.payload.get("tipo_documento", ""),
                      reference=reference(p.payload), source=p.payload.get("fuente", ""),
                      text=p.payload.get("texto", ""), payload=dict(p.payload))
               for i, p in enumerate(points, 1)]
    return sources, (t1 - t0) * 1000, (t2 - t1) * 1000


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_context(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.n}] {s.label}\n{s.text.strip()}" for s in sources)


def build_prompt(question: str, sources: list[Source], history: list[dict] | None = None) -> str:
    parts = []
    if history:
        convo = "\n".join(f"{'Cliente' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
                          for m in history[-2 * HISTORY_TURNS:])
        parts.append(f"CONVERSACIÓN PREVIA (solo para entender el contexto, no es una fuente):\n{convo}")
    parts.append(f"FUENTES:\n{build_context(sources)}")
    parts.append(f"PREGUNTA DEL CLIENTE: {question}\n\n"
                 "Responde siguiendo las reglas, con citas [n] de las FUENTES.")
    return "\n\n".join(parts)


REWRITE_SYSTEM = ("Reescribes la última pregunta de un cliente como una consulta de búsqueda autónoma y completa en español. "
                  "Usa la conversación previa solo para resolver referencias como 'eso', 'y si…', 'ese producto'. "
                  "Usa términos de la política de devoluciones de la tienda cuando aplique "
                  "(por ejemplo: producto llega dañado o incompleto, derecho de retracto, devolución, garantía, "
                  "reversión del pago, envío internacional, aranceles, precio del producto). "
                  "Máximo 15 palabras. No menciones leyes ni artículos que el cliente no nombró. "
                  "No respondas la pregunta. Devuelve solo la consulta, sin explicaciones.")


def rewrite_question(question: str, history: list[dict]) -> str:
    """Standalone search query: resolves follow-ups and maps colloquial words to the documents' vocabulary
    ('me arrepentí' -> 'derecho de retracto')."""
    convo = "\n".join(f"{'Cliente' if m['role'] == 'user' else 'Asistente'}: {m['content'][:400]}"
                      for m in (history or [])[-2 * HISTORY_TURNS:])
    prompt = (f"Conversación previa:\n{convo}\n\n" if convo else "") + f"Última pregunta: {question}\n\nConsulta:"
    try:
        out = llm(prompt, system=REWRITE_SYSTEM, temperature=0.0, max_tokens=128)
    except RuntimeError:
        return question
    return out.strip().strip('"') or question


def retrieve_multi(question: str, search_query: str, k: int, doc_types=None):
    """Searches with the original question and the rewritten query, keeps the best score per chunk."""
    a, e1, q1 = retrieve(question, k, doc_types)
    if search_query == question:
        return a, e1, q1
    b, e2, q2 = retrieve(search_query, k, doc_types)
    best = {}
    for s in a + b:
        key = s.text[:200]
        if key not in best or s.score > best[key].score:
            best[key] = s
    merged = sorted(best.values(), key=lambda s: s.score, reverse=True)[:k]
    for i, s in enumerate(merged, 1):
        s.n = i
    return merged, e1 + e2, q1 + q2


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def cited_numbers(text: str, n_sources: int) -> list[int]:
    nums = {int(x) for x in re.findall(r"\[(\d+)\]", text)}
    return sorted(n for n in nums if 1 <= n <= n_sources)


def ask(question: str, history: list[dict] | None = None, k: int = TOP_K,
        doc_types: list[str] | None = None) -> Answer:
    t_rw = time.perf_counter()
    search_query = rewrite_question(question, history or [])
    ms_rewrite = (time.perf_counter() - t_rw) * 1000

    sources, ms_emb, ms_q = retrieve_multi(question, search_query, k, doc_types)
    t0 = time.perf_counter()
    text = llm(build_prompt(question, sources, history))
    ms_llm = (time.perf_counter() - t0) * 1000 + ms_rewrite
    refused = NOT_FOUND.lower().rstrip(".") in text.lower()
    return Answer(question=question, search_query=search_query, text=text, sources=sources,
                  cited=[] if refused else cited_numbers(text, len(sources)),
                  ms_embedding=ms_emb, ms_qdrant=ms_q, ms_llm=ms_llm, refused=refused)


# ---------------------------------------------------------------------------
# Document management helpers
# ---------------------------------------------------------------------------
def collection_info() -> dict:
    info = qdrant().get_collection(COLLECTION)
    vec = info.config.params.vectors
    return {"status": str(info.status).split(".")[-1].lower(), "points": info.points_count,
            "dimension": vec.size, "distance": str(vec.distance).split(".")[-1]}


def all_chunks() -> list[dict]:
    """Every point's payload (no vectors). 555 points -> a few scroll pages."""
    out, offset = [], None
    while True:
        points, offset = qdrant().scroll(COLLECTION, limit=256, offset=offset,
                                         with_payload=True, with_vectors=False)
        out.extend({"id": str(p.id), **p.payload} for p in points)
        if offset is None:
            return out


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "¿Cuántos días tengo para retractarme de una compra por internet?"
    a = ask(q)
    print(a.text, "\n")
    for s in a.sources:
        print(f"[{s.n}] {s.score:.3f} {s.label}")
    print(f"\nLatencia: embedding {a.ms_embedding:.0f} ms · Qdrant {a.ms_qdrant:.0f} ms · Gemini {a.ms_llm:.0f} ms")
