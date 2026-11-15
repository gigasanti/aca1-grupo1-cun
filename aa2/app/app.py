"""ACA 2 · Aprendizaje Automático · Grupo 1 CUN
Pronóstico de ventas diarias (UCI Online Retail) con una red LSTM entrenada en el notebook ACA2_AA_LSTM_Ventas_Grupo1.ipynb.
"""
import json
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
AZUL, NARANJA, GRIS = "#1f4e79", "#e3a33c", "#9AA39E"
DIAS = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}

st.set_page_config(page_title="Pronóstico de ventas con LSTM", page_icon="📈", layout="wide")


@st.cache_resource
def cargar_modelo():
    import tensorflow as tf
    return tf.keras.models.load_model(BASE / "modelo_lstm.keras")


@st.cache_data
def cargar_datos():
    cfg = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    diaria = pd.read_csv(BASE / "ventas_diarias.csv", parse_dates=["fecha"])
    prueba = pd.read_csv(BASE / "prueba_pronosticos.csv", parse_dates=["fecha"])
    return cfg, diaria, prueba


def proximos_dias_con_venta(desde, n, dias_validos):
    fechas, f = [], desde
    while len(fechas) < n:
        f = f + pd.Timedelta(days=1)
        if f.dayofweek in dias_validos:
            fechas.append(f)
    return fechas


def pronosticar(modelo, historia, fechas_hist, horizonte, cfg):
    """Pronóstico recursivo: cada día pronosticado entra a la ventana para pronosticar el siguiente.
    Mismo preprocesamiento del notebook: log1p, resta del promedio de la ventana y día de la semana one-hot."""
    ventana, validos = cfg["ventana"], cfg["dias_con_venta"]
    y = list(np.log1p(historia[-ventana:]))
    dows = list(fechas_hist[-ventana:].dt.dayofweek)
    futuras = proximos_dias_con_venta(fechas_hist.iloc[-1], horizonte, validos)
    salida = []
    for f in futuras:
        base = np.mean(y[-ventana:])
        dias_sig = dows[-ventana + 1:] + [f.dayofweek]
        onehot = np.array([[1.0 if d == v else 0.0 for v in validos] for d in dias_sig])
        x = np.column_stack([np.array(y[-ventana:]) - base, onehot])[None, ...]
        pred_log = float(modelo.predict(x, verbose=0)[0, 0]) + base
        salida.append(np.expm1(pred_log))
        y.append(pred_log)
        dows.append(f.dayofweek)
    return pd.DataFrame({"fecha": futuras, "pronostico": salida})


def libras(v):
    return f"£{v:,.0f}".replace(",", ".")


cfg, diaria, prueba = cargar_datos()
m = cfg["metricas"]
base_nombre = min((k for k in m if k not in ("LSTM", "GRU")), key=lambda k: m[k]["MAE (£)"])
mejora = 100 * (1 - m["LSTM"]["MAE (£)"] / m[base_nombre]["MAE (£)"])

st.title("📈 Pronóstico de ventas diarias con una red LSTM")
st.caption("ACA 2 · Aprendizaje Automático · Especialización en IA (CUN) · Grupo 1 · Dataset UCI Online Retail (Chen, 2015)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Error medio diario (MAE)", libras(m["LSTM"]["MAE (£)"]), f"-{mejora:.1f}% vs mejor método simple", delta_color="inverse")
k2.metric("Error porcentual (MAPE)", f"{m['LSTM']['MAPE (%)']:.1f}%", f"{m['LSTM']['MAPE (%)'] - m['Ingenuo estacional (semana pasada)']['MAPE (%)']:.1f} pts vs semana pasada", delta_color="inverse")
k3.metric("Acierto en dirección (F1)", f"{cfg['direccion']['f1_macro']:.2f}", f"accuracy {cfg['direccion']['accuracy']:.0%}", delta_color="off")
k4.metric("Error del total semanal (mediana)", f"{cfg['semanal']['lstm_mediana_pct']:.1f}%", f"vs {cfg['semanal']['semana_pasada_mediana_pct']:.1f}% repitiendo la semana pasada", delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Pronóstico", "📊 Evaluación del modelo", "🗄️ Datos (SQL)", "🧠 Cómo funciona"])

with tab1:
    izq, der = st.columns([1, 3])
    with izq:
        st.subheader("Configura el pronóstico")
        opciones = list(diaria["fecha"].dt.date.iloc[cfg["ventana"]:])
        st.session_state.setdefault("corte", pd.Timestamp("2011-11-21").date())
        st.caption("Ejemplos para la demo")
        e1, e2 = st.columns(2)
        e1.button("Semana del 21 nov", width="stretch", on_click=lambda: st.session_state.update(corte=pd.Timestamp("2011-11-21").date()))
        e2.button("Último dato (futuro)", width="stretch", on_click=lambda: st.session_state.update(corte=opciones[-1]))
        corte = st.select_slider(
            "Pronosticar desde el día…", options=opciones, key="corte",
            help="El modelo solo ve las ventas hasta este día. Si eliges una fecha pasada puedes comparar contra lo que se vendió de verdad.",
        )
        horizonte = st.slider("Días a pronosticar", 1, 14, 7, help="Solo cuenta días con venta: la tienda no vende los sábados.")

    hist = diaria[diaria["fecha"].dt.date <= corte]
    with st.spinner("La LSTM está calculando…"):
        pron = pronosticar(cargar_modelo(), hist["ventas"].to_numpy(), hist["fecha"], horizonte, cfg)
    real_fut = diaria.merge(pron[["fecha"]], on="fecha")

    with der:
        st.subheader(f"Pronóstico de {horizonte} días desde el {pd.Timestamp(corte):%d %b %Y}")
        vista = pd.concat([
            hist.tail(42).assign(serie="Histórico")[["fecha", "ventas", "serie"]],
            real_fut.assign(serie="Real (lo que pasó)")[["fecha", "ventas", "serie"]],
            pron.rename(columns={"pronostico": "ventas"}).assign(serie="Pronóstico LSTM"),
        ])
        colores = alt.Scale(domain=["Histórico", "Real (lo que pasó)", "Pronóstico LSTM"], range=[GRIS, "black", AZUL])
        graf = alt.Chart(vista).mark_line(point=True).encode(
            x=alt.X("fecha:T", title=None),
            y=alt.Y("ventas:Q", title="Ventas (£)"),
            color=alt.Color("serie:N", scale=colores, legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.condition(alt.datum.serie == "Pronóstico LSTM", alt.value([6, 3]), alt.value([1, 0])),
            tooltip=[alt.Tooltip("fecha:T", format="%a %d %b %Y"), "serie:N", alt.Tooltip("ventas:Q", format=",.0f")],
        ).properties(height=340)
        st.altair_chart(graf, width="stretch")

        tabla = pron.copy()
        tabla["día"] = tabla["fecha"].dt.dayofweek.map(DIAS)
        tabla = tabla.merge(diaria[["fecha", "ventas"]], on="fecha", how="left").rename(columns={"ventas": "real"})
        c1, c2, c3 = st.columns(3)
        c1.metric("Venta pronosticada del periodo", libras(tabla["pronostico"].sum()))
        if tabla["real"].notna().all():
            err = 100 * (tabla["pronostico"].sum() - tabla["real"].sum()) / tabla["real"].sum()
            c2.metric("Venta real del periodo", libras(tabla["real"].sum()))
            c3.metric("Error del total", f"{err:+.1f}%")
        else:
            c2.info("Fechas posteriores al último dato: no hay venta real para comparar.")
        mostrar = tabla[["fecha", "día", "pronostico", "real"]].copy()
        mostrar["fecha"] = mostrar["fecha"].dt.date
        st.dataframe(mostrar.style.format({"pronostico": "£{:,.0f}", "real": "£{:,.0f}"}, na_rep="—"), hide_index=True, width="stretch")

with tab2:
    st.subheader("Prueba: últimos 60 días (temporada navideña 2011)")
    tm = pd.DataFrame(m).T[["MAE (£)", "RMSE (£)", "MAPE (%)"]].sort_values("MAE (£)")
    a, b = st.columns([2, 3])
    with a:
        st.markdown("**Error por modelo** (menor es mejor)")
        st.dataframe(tm.style.format("{:,.1f}").highlight_min(color="#d6e9d8"), width="stretch")
        st.markdown("**Dirección del día siguiente: ¿sube o baja?**")
        cm = np.array(cfg["direccion"]["matriz"])
        cm_df = pd.DataFrame([(r, p, int(cm[i, j])) for i, r in enumerate(["Baja", "Sube"]) for j, p in enumerate(["Baja", "Sube"])],
                             columns=["Real", "Pronóstico", "Días"])
        base_cm = alt.Chart(cm_df).encode(x=alt.X("Pronóstico:N"), y=alt.Y("Real:N"))
        st.altair_chart((base_cm.mark_rect().encode(color=alt.Color("Días:Q", scale=alt.Scale(scheme="blues"), legend=None))
                         + base_cm.mark_text(fontSize=18).encode(text="Días:Q")).properties(height=180, width=220))
        st.caption(f"F1 macro {cfg['direccion']['f1_macro']:.2f} · accuracy {cfg['direccion']['accuracy']:.0%}")
    with b:
        largo = prueba.melt(id_vars="fecha", value_vars=["real", "LSTM", "Ingenuo estacional (semana pasada)"], var_name="serie", value_name="ventas")
        largo["serie"] = largo["serie"].replace({"real": "Real", "Ingenuo estacional (semana pasada)": "Semana pasada"})
        st.altair_chart(alt.Chart(largo).mark_line().encode(
            x=alt.X("fecha:T", title=None), y=alt.Y("ventas:Q", title="Ventas (£)"),
            color=alt.Color("serie:N", scale=alt.Scale(domain=["Real", "LSTM", "Semana pasada"], range=["black", AZUL, NARANJA]), legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("fecha:T", format="%d %b"), "serie:N", alt.Tooltip("ventas:Q", format=",.0f")],
        ).properties(height=300), width="stretch")
        curvas = pd.DataFrame({"época": range(1, len(cfg["curvas"]["loss"]) + 1), "Entrenamiento": cfg["curvas"]["loss"], "Validación": cfg["curvas"]["val_loss"]}).melt("época", var_name="serie", value_name="pérdida")
        st.markdown("**Curvas de aprendizaje (pérdida Huber)**")
        st.altair_chart(alt.Chart(curvas).mark_line().encode(
            x="época:Q", y=alt.Y("pérdida:Q", title="Pérdida"),
            color=alt.Color("serie:N", scale=alt.Scale(range=[AZUL, NARANJA]), legend=alt.Legend(orient="top", title=None)),
        ).properties(height=200), width="stretch")

with tab3:
    st.subheader("Consulta la serie diaria con SQL")
    st.caption("Tabla `ventas_diarias` (fecha, ventas, pedidos) en una base SQLite en memoria. Solo se permiten consultas SELECT.")
    ejemplos = {
        "Venta media por día de la semana": "SELECT strftime('%w', fecha) AS dia_semana, COUNT(*) AS dias, ROUND(AVG(ventas)) AS venta_media\nFROM ventas_diarias GROUP BY dia_semana ORDER BY dia_semana",
        "Ventas por mes": "SELECT substr(fecha, 1, 7) AS mes, ROUND(SUM(ventas)) AS ventas, SUM(pedidos) AS pedidos\nFROM ventas_diarias GROUP BY mes ORDER BY mes",
        "Los 10 días de mayor venta": "SELECT fecha, ROUND(ventas) AS ventas, pedidos FROM ventas_diarias\nORDER BY ventas DESC LIMIT 10",
    }
    elegido = st.selectbox("Consulta de ejemplo", list(ejemplos))
    consulta = st.text_area("SQL", ejemplos[elegido], height=110)
    if st.button("Ejecutar consulta"):
        if not consulta.strip().lower().startswith("select"):
            st.error("Solo se permiten consultas SELECT.")
        else:
            con = sqlite3.connect(":memory:")
            diaria.assign(fecha=diaria["fecha"].dt.strftime("%Y-%m-%d")).to_sql("ventas_diarias", con, index=False)
            try:
                st.dataframe(pd.read_sql(consulta, con), hide_index=True, width="stretch")
            except Exception as e:
                st.error(f"Error en la consulta: {e}")
            finally:
                con.close()

with tab4:
    st.subheader("Del dato a la decisión")
    st.markdown(f"""
1. **SQL (SQLite):** 541.909 líneas de factura → limpieza (sin cancelaciones, cantidades ni precios negativos) → **venta diaria total** (305 días).
2. **Preparación:** escala logarítmica, resta del promedio de los últimos {cfg['ventana']} días (la red aprende la *forma*, no el nivel) y día de la semana.
3. **Red LSTM:** lee {cfg['ventana']} días de historia y pronostica el día siguiente. Para varios días, cada pronóstico entra a la ventana del siguiente.
4. **Evaluación:** últimos 60 días (oct – dic 2011) contra tres métodos simples. La LSTM tuvo el menor error diario y semanal.

| Capa | Configuración |
|---|---|
| Entrada | {cfg['ventana']} días × 7 variables |
| LSTM | 32 unidades |
| Dropout | 0,2 |
| Densa | 16, ReLU |
| Salida | 1 (venta del día siguiente) |

Adam (0,001) · pérdida Huber · parada temprana · {cfg['epocas']} épocas.

**Limitación:** con 13 meses de historia el modelo ve una sola Navidad y no anticipa pedidos excepcionales.
""")
    st.caption("Código y notebook: github.com/gigasanti/aca1-grupo1-cun (carpeta aa2/)")
