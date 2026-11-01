# ACA 1 · Grupo 1 · Especialización en Inteligencia Artificial (CUN)

Pilar **Ventas** · Dataset base **UCI Online Retail** (Chen, 2015) · Docente: Dr. Hamilton Rivera Flor

Integrantes: Mayra Fernanda Bernate Bustamante · Angela Consuelo Castillo Rodríguez · Karen Viviana Ospina Lesmes · Jair Vargas · Santiago Bahamón Escobar

| Carpeta | Asignatura | Contenido |
|---|---|---|
| [`nube/`](nube) | Computación en la Nube para Sistemas Inteligentes | Pipeline RAG: ingesta, chunking, embeddings (`multilingual-e5-small`) y búsqueda semántica en **Qdrant Cloud** |
| [`aa/`](aa) | Aprendizaje Automático | Segmentación de clientes con **RFM + K-Means** y app interactiva en **Streamlit** |

## Computación en la Nube (`nube/`)

1. Copiar `.env.example` a `.env` y completar `QDRANT_URL` y `QDRANT_API_KEY`. En Google Colab se pueden usar los *Secrets*.
2. `pip install -r requirements.txt`
3. Ejecutar `ACA1_RAG_Ventas_Grupo1.ipynb` de principio a fin. El notebook descarga el dataset, la Ley 1480 de 2011 y el TFG de marketing desde sus fuentes públicas.

Corpus complementario: Ley 1480 de 2011 (Estatuto del Consumidor), política propia de devoluciones, garantías y SLA, y el TFG *Optimización de campañas de marketing mediante análisis predictivo y segmentación de clientes* (Universidad de Jaén).

## Aprendizaje Automático (`aa/`)

- Notebook: `ACA1_AA_Online_Retail_Grupo1.ipynb` (EDA, limpieza, RFM, K-Means, evaluación y exportación del modelo). Requiere `online+retail/Online Retail.xlsx`, descargable de https://archive.ics.uci.edu/dataset/352/online+retail.
- App: `app/` (`app.py`, `modelo_rfm.pkl`, `requirements.txt`).

```
cd aa/app
pip install -r requirements.txt
streamlit run app.py
```

App publicada: https://aca1-grupo1-cun-jqvtqkdarfsifdxeqfwhjv.streamlit.app

## Videos de sustentación

- Computación en la Nube: _[enlace]_
- Aprendizaje Automático: https://youtu.be/o83qOFIDKzw
