"""Rebuilds the ACA 1 vector collection (ventas_rag_aca1) in Qdrant Cloud from the corpus files.

Use it only if the collection is missing (e.g. a free-tier cluster was deleted after inactivity).
It reproduces the ACA 1 pipeline: same cleaning, chunking (900 chars, overlap 150), e5-small
"passage:" embeddings, payload fields and deterministic UUID v5 ids, so the result is the same
555 points the ACA 1 notebook uploaded.

    python indexar.py            # creates the collection if it does not exist
    python indexar.py --force    # deletes and recreates it
"""
import re
import sys
import time
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

import rag

CORPUS = Path(__file__).resolve().parent / "corpus"
URLS = {
    "ley_1480_2011_estatuto_consumidor.pdf": "https://www.funcionpublica.gov.co/eva/gestornormativo/norma_pdf.php?i=44306",
    "tfg_marketing_segmentacion.pdf": "https://crea.ujaen.es/server/api/core/bitstreams/70acee9f-e7ba-428d-b21d-507a2b75f91e/content",
}
FUENTE = {
    "normativa": "Ley 1480 de 2011 (Congreso de Colombia, 2011)",
    "politica_comercial": "Política de devoluciones, garantías y SLA (Grupo 1, 2026)",
    "literatura": "El Jannus (2024), TFG Universidad de Jaén",
    "ficha_producto": "UCI Online Retail (Chen, 2015)",
}


def ensure(name: str) -> Path:
    p = CORPUS / name
    if not p.exists():
        r = requests.get(URLS[name], timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


def pages(path: Path):
    with fitz.open(path) as doc:
        return [(i + 1, pg.get_text("text")) for i, pg in enumerate(doc)]


def clean(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def records():
    out = []
    for pag, txt in pages(ensure("ley_1480_2011_estatuto_consumidor.pdf")):
        txt = re.sub(r"(?im)^\s*(Departamento Administrativo de la Función Pública|Ley 1480 de 2011|EVA - Gestor Normativo|\d{1,3})\s*$", "", txt)
        out.append((clean(txt), {"tipo_documento": "normativa", "fuente": FUENTE["normativa"], "pagina": pag}))

    politica = (CORPUS / "politica_devoluciones_garantias_sla.md").read_text(encoding="utf-8")
    for bloque in re.split(r"\n(?=## )", politica):
        titulo = bloque.splitlines()[0].lstrip("# ").strip()
        partes = re.split(r"\n(?=\d+\.\d+\. )", bloque)
        if len(partes) > 1 and len(clean(partes[0])) <= len(titulo) + 10:
            partes = partes[1:]
        for parte in partes:
            numeral = re.match(r"\d+\.\d+", parte.strip())
            seccion = f"{titulo} · {numeral.group(0)}" if numeral else titulo
            texto = f"Política de devoluciones de la tienda · {titulo}\n{parte}" if numeral else parte
            out.append((clean(texto), {"tipo_documento": "politica_comercial", "fuente": FUENTE["politica_comercial"], "seccion": seccion}))

    for pag, txt in pages(ensure("tfg_marketing_segmentacion.pdf")):
        if pag >= 107:  # bibliography
            continue
        txt = re.sub(r"El Jannus,\s*Salma\s+Optimización de campañas de marketing mediante\s+análisis predictivo y segmentación de clientes", "", txt, flags=re.I)
        txt = re.sub(r"Escuela Politécnica Superior de Jaén", "", txt, flags=re.I)
        out.append((clean(txt), {"tipo_documento": "literatura", "fuente": FUENTE["literatura"], "pagina": pag}))
    return [(t, m) for t, m in out if len(t) > 80]


def fichas():
    """Product sheets generated in ACA 1 from UCI Online Retail (one sheet = one chunk)."""
    txt = (CORPUS / "fichas_productos_online_retail.md").read_text(encoding="utf-8")
    out = []
    for f in re.findall(r"^Ficha de producto .+$", txt, flags=re.M):
        m = re.match(r"Ficha de producto (\S+): (.+?)\. Categoría: (.+?)\. Precio unitario de referencia: ([\d.]+)", f)
        pais = re.search(r"País con más ventas: (.+?)\.", f)
        out.append((f, {"tipo_documento": "ficha_producto", "fuente": FUENTE["ficha_producto"],
                        "stock_code": m.group(1), "descripcion": m.group(2), "categoria": m.group(3),
                        "precio_medio": float(m.group(4)), "pais_principal": pais.group(1) if pais else None}))
    return out


def point_id(meta: dict) -> str:
    key = f"{meta['fuente']}|{meta.get('pagina', meta.get('seccion', meta.get('stock_code')))}|{meta['chunk']}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def main(force: bool = False):
    client = rag.qdrant()
    if client.collection_exists(rag.COLLECTION) and not force:
        print(f"La colección {rag.COLLECTION} ya existe con {client.count(rag.COLLECTION, exact=True).count} puntos. Use --force para recrearla.")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])
    chunks, metas = [], []
    for text, meta in records():
        for i, frag in enumerate(f for f in splitter.split_text(text) if len(f.strip()) >= 50):
            chunks.append(frag)
            metas.append({**meta, "chunk": i})
    for text, meta in fichas():
        chunks.append(text)
        metas.append({**meta, "chunk": 0})
    print("Fragmentos:", len(chunks))

    vectors = rag.embedder().encode([f"passage: {c}" for c in chunks], batch_size=32,
                                    normalize_embeddings=True, show_progress_bar=True)
    if client.collection_exists(rag.COLLECTION):
        client.delete_collection(rag.COLLECTION)
    client.create_collection(rag.COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    for campo in ["tipo_documento", "fuente", "pais_principal"]:
        client.create_payload_index(rag.COLLECTION, field_name=campo, field_schema=PayloadSchemaType.KEYWORD)
    pts = [PointStruct(id=point_id(m), vector=v.tolist(), payload={**m, "texto": t})
           for t, m, v in zip(chunks, metas, vectors)]
    t0 = time.perf_counter()
    for i in range(0, len(pts), 64):
        client.upsert(rag.COLLECTION, points=pts[i:i + 64], wait=True)
    print(f"Upsert {len(pts)} puntos en {time.perf_counter() - t0:.1f} s · total en la nube: "
          f"{client.count(rag.COLLECTION, exact=True).count}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
