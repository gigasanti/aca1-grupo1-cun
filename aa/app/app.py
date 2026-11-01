from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Segmentación de clientes RFM", page_icon="🛒", layout="wide")

COLORES = {
    "Campeones": "#1B7A3E",
    "Clientes leales": "#7FBF8F",
    "Nuevos / recientes": "#E3B23C",
    "Perdidos / inactivos": "#9AA39E",
}
RECOMENDACIONES = {
    "Campeones": "Compran seguido, hace poco y gastan mucho. Cuidarlos: programa de recompensas y acceso anticipado a productos.",
    "Clientes leales": "Compran con regularidad, aunque no tan reciente. Ventas cruzadas para aumentar el valor de cada pedido.",
    "Nuevos / recientes": "Compraron hace poco, pero pocas veces. Ofertas de segunda compra para volverlos clientes leales.",
    "Perdidos / inactivos": "Compraron una o dos veces y hace mucho tiempo. Campañas de bajo costo o dejar de invertir en ellos.",
}


@st.cache_resource
def cargar_modelo():
    return joblib.load(Path(__file__).parent / "modelo_rfm.pkl")


def clasificar(modelo, recencia, frecuencia, monetario):
    """Aplica el mismo preprocesamiento del notebook: log1p + StandardScaler + K-Means."""
    entrada = pd.DataFrame([[recencia, frecuencia, monetario]], columns=modelo["columnas"])
    escalada = modelo["scaler"].transform(np.log1p(entrada))
    distancias = modelo["kmeans"].transform(escalada)[0]
    cercania = (1 / distancias) / (1 / distancias).sum()
    parecido = pd.DataFrame({
        "Segmento": [modelo["mapa_nombres"][i] for i in range(len(distancias))],
        "Parecido": cercania * 100,
    })
    return modelo["mapa_nombres"][int(distancias.argmin())], parecido


def clientes_tipo(modelo):
    """Cliente típico de cada segmento: el centro del grupo, devuelto a unidades originales."""
    centros = np.expm1(modelo["scaler"].inverse_transform(modelo["kmeans"].cluster_centers_))
    return {
        modelo["mapa_nombres"][i]: {"dias": round(r), "compras": max(1, round(f)), "ticket": int(round(m / f, -1))}
        for i, (r, f, m) in enumerate(centros)
    }


def barras(df, valor, titulo, formato):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Segmento:N", sort=list(COLORES), title=None, axis=alt.Axis(labelLimit=200, labelFontSize=13, labelOverlap=False)),
            x=alt.X(f"{valor}:Q", title=titulo),
            color=alt.Color("Segmento:N", scale=alt.Scale(domain=list(COLORES), range=list(COLORES.values())), legend=None),
            tooltip=["Segmento", alt.Tooltip(f"{valor}:Q", format=formato)],
        )
        .properties(height=190)
    )


try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró `modelo_rfm.pkl`. Ponlo en la misma carpeta que `app.py`.")
    st.stop()
TIPO = clientes_tipo(modelo)

st.markdown("<style>.block-container{padding-top:2.2rem;padding-bottom:1rem;max-width:1300px}</style>", unsafe_allow_html=True)
st.markdown("## 🛒 Segmentación de clientes")
st.caption(
    "Un modelo de Machine Learning (K-Means) agrupó a los 4.333 clientes de una tienda en línea (dataset UCI Online Retail) "
    "según **cuándo**, **cuántas veces** y **cuánto** compran (RFM). Describe un cliente y la app dice a qué grupo se parece más."
)

tab1, tab2 = st.tabs(["🔎 Clasificar un cliente", "📊 Resumen de segmentos"])

with tab1:
    entrada, salida = st.columns([1, 1.15], gap="large")

    with entrada:
        st.markdown("**1. Empieza con un cliente de ejemplo** (o mueve los controles):")


        def cargar_ejemplo(segmento):
            st.session_state.update(TIPO[segmento])


        fila1, fila2 = st.columns(2), st.columns(2)
        for col, segmento in zip(fila1 + fila2, COLORES):
            col.button(segmento, on_click=cargar_ejemplo, args=(segmento,), width="stretch")

        st.markdown("**2. Describe al cliente:**")
        st.session_state.setdefault("dias", 30)
        st.session_state.setdefault("compras", 3)
        st.session_state.setdefault("ticket", 300)
        dias = st.slider("¿Hace cuántos días fue su última compra?", 0, 365, key="dias")
        compras = st.slider("¿Cuántas compras ha hecho en el año?", 1, 60, key="compras")
        ticket = st.slider("¿Cuánto gasta en promedio en cada compra? (£)", 10, 2000, step=10, key="ticket")
        gasto = compras * ticket
        st.caption(f"Gasto total en el año: **£{gasto:,.0f}** ({compras} compras × £{ticket:,}).")

    segmento, parecido = clasificar(modelo, dias, compras, gasto)
    tipo = TIPO[segmento]
    with salida:
        st.markdown("**3. Resultado:**")
        st.success(f"### {segmento}\n{RECOMENDACIONES[segmento]}")
        st.markdown(f"**¿Por qué?** El modelo compara las tres cifras con el cliente típico de cada grupo y elige el más cercano:")
        st.table(pd.DataFrame(
            {
                "Este cliente": [f"hace {dias} días", f"{compras}", f"£{gasto:,.0f}"],
                f"Típico de {segmento}": [f"hace {tipo['dias']} días", f"{tipo['compras']}", f"£{tipo['compras'] * tipo['ticket']:,.0f}"],
            },
            index=["Última compra", "Compras en el año", "Gasto total"],
        ))
        st.altair_chart(barras(parecido, "Parecido", "Parecido con cada grupo (%)", ".0f").properties(height=165), width="stretch")

with tab2:
    resumen = pd.DataFrame(modelo["resumen"])
    resumen["% de clientes"] = resumen["Clientes"] / resumen["Clientes"].sum() * 100
    ingreso = resumen["Clientes"] * resumen["Gasto_medio"]
    resumen["% del ingreso"] = ingreso / ingreso.sum() * 100
    campeones = resumen.set_index("Segmento").loc["Campeones"]
    st.info(
        f"Los **Campeones** son el {campeones['% de clientes']:.0f} % de los clientes "
        f"y generan el {campeones['% del ingreso']:.0f} % del ingreso."
    )

    izq, der = st.columns(2, gap="large")
    izq.altair_chart(barras(resumen, "% de clientes", "% de clientes", ".0f"), width="stretch")
    der.altair_chart(barras(resumen, "% del ingreso", "% del ingreso", ".0f"), width="stretch")

    st.markdown("**Cliente promedio de cada grupo**")
    tabla = resumen.set_index("Segmento")[["Clientes", "Recencia_media", "Frecuencia_media", "Gasto_medio"]]
    tabla.columns = ["Clientes", "Días desde última compra", "Compras en el año", "Gasto total (£)"]
    st.dataframe(
        tabla.style.format({"Días desde última compra": "{:.0f}", "Compras en el año": "{:.1f}", "Gasto total (£)": "£{:,.0f}"}),
        width="stretch",
    )
