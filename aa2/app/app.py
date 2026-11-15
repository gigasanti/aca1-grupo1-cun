"""ACA 2 · Aprendizaje Automático · Grupo 1 CUN
Pronóstico de ventas (UCI Online Retail) con una red LSTM entrenada en el notebook ACA2_AA_LSTM_Ventas_Grupo1.ipynb.
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
AZUL, GRIS, GRIS_OSC = "#3b7dd8", "#9aa3ad", "#6b737c"
DIAS = ["Dom", "Lun", "Mar", "Mié", "Jue", "Vie"]          # la tienda no vende los sábados
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
HORIZONTES = {"Semana": 1, "Mes": 4, "Trimestre": 13}
ETIQ_FECHA = "date(datum.value) + ' ' + ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'][month(datum.value)]"

st.set_page_config(page_title="Pronóstico de ventas con LSTM", page_icon="📈", layout="wide")


# ---------- utilidades ----------
def miles(v):
    return f"{v:,.0f}".replace(",", ".")


def libras(v):
    return f"£{miles(v)}"


def libras_mil(v):
    return f"£{v / 1000:,.0f} mil".replace(",", ".")


def corta(f):
    return f"{f.day} {MESES[f.month - 1][:3]}"


def domingo_de(f):
    f = pd.Timestamp(f)
    return f - pd.Timedelta(days=(f.dayofweek + 1) % 7)


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


def pronosticar(modelo, historia, fechas_hist, fechas_obj, cfg):
    """Pronóstico recursivo día a día: cada pronóstico entra a la ventana del siguiente.
    Mismo preprocesamiento del notebook: log1p, resta del promedio de la ventana y día de la semana one-hot."""
    ventana, validos = cfg["ventana"], cfg["dias_con_venta"]
    y = list(np.log1p(historia[-ventana:]))
    dows = list(fechas_hist[-ventana:].dt.dayofweek)
    salida = []
    for f in fechas_obj:
        base = np.mean(y[-ventana:])
        dias_sig = dows[-ventana + 1:] + [f.dayofweek]
        onehot = np.array([[1.0 if d == v else 0.0 for v in validos] for d in dias_sig])
        x = np.column_stack([np.array(y[-ventana:]) - base, onehot])[None, ...]
        pred = float(modelo.predict(x, verbose=0)[0, 0]) + base
        salida.append(np.expm1(pred))
        y.append(pred)
        dows.append(f.dayofweek)
    return np.array(salida)


cfg, diaria, prueba = cargar_datos()
m = cfg["metricas"]
hz = cfg["horizontes"]
inicio_prueba = diaria["fecha"].iloc[-cfg["dias_prueba"]]
ultimo_dato = diaria["fecha"].iloc[-1]
primer_inicio = domingo_de(diaria["fecha"].iloc[cfg["ventana"]]) + pd.Timedelta(weeks=1)
base_mae = min(v["MAE (£)"] for k, v in m.items() if k not in ("LSTM", "GRU"))


def inicio_valido(inicio, semanas):
    fin = inicio + pd.Timedelta(weeks=semanas) - pd.Timedelta(days=1)
    return primer_inicio <= inicio and fin <= ultimo_dato + pd.Timedelta(days=1)


def ultimo_inicio_valido(semanas):
    s = domingo_de(ultimo_dato)
    while not inicio_valido(s, semanas):
        s -= pd.Timedelta(weeks=1)
    return s


# ---------- encabezado ----------
st.title("📈 Pronóstico de ventas con una red LSTM")
st.caption("ACA 2 · Aprendizaje Automático · Especialización en IA (CUN) · Grupo 1 · Tienda de regalos en línea de Reino Unido, dic 2010 – dic 2011 (Chen, 2015)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Error diario promedio", libras_mil(m["LSTM"]["MAE (£)"]), f"{100 * (m['LSTM']['MAE (£)'] / base_mae - 1):.0f}% vs mejor método simple", delta_color="inverse",
          help="En promedio, cada día el pronóstico se aleja esa cantidad de la venta real, por encima o por debajo (MAE). Últimos 60 días con venta.")
k2.metric("Error diario en %", f"{m['LSTM']['MAPE (%)']:.0f}%".replace(".", ","), f"{m['LSTM']['MAPE (%)'] - m['Ingenuo estacional (semana pasada)']['MAPE (%)']:.0f} pts vs repetir semana pasada", delta_color="inverse",
          help="El mismo error, como porcentaje de la venta de cada día (MAPE). Si un día se venden £50 mil, un error de 25% es £12,5 mil.")
k3.metric("Acierta si mañana sube o baja", f"{round(cfg['direccion']['accuracy'] * 10):.0f} de 10 días",
          help="¿Mañana se vende más o menos que hoy? La red acierta 78% de los días. El F1 promedia los aciertos en 'sube' y en 'baja' para que no gane solo por adivinar el caso más común.")
k4.metric("Error típico al planear una semana", f"{hz['Semana']['lstm']:.0f}%", f"{hz['Semana']['lstm'] - hz['Semana']['anterior']:.0f} pts vs repetir la semana anterior", delta_color="inverse",
          help="Mediana del error del total de la semana (domingo a viernes): en la mitad de las semanas la diferencia, por encima o por debajo, fue menor que este valor.")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Pronóstico", "📊 Evaluación del modelo", "🗄️ Datos (SQL)", "🧠 Cómo funciona"])

# ---------- pestaña 1: pronóstico ----------
with tab1:
    st.session_state.setdefault("inicio", pd.Timestamp("2011-11-27"))
    izq, der = st.columns([1, 2.3], gap="large")

    with izq:
        st.markdown("#### ¿Qué quieres planear?")
        horizonte = st.segmented_control("Horizonte", list(HORIZONTES), default="Semana", key="horizonte", label_visibility="collapsed") or "Semana"
        semanas = HORIZONTES[horizonte]
        if not inicio_valido(st.session_state["inicio"], semanas):
            st.session_state["inicio"] = ultimo_inicio_valido(semanas)
            st.session_state.pop("mes", None)
        inicio = st.session_state["inicio"]
        fin = inicio + pd.Timedelta(weeks=semanas) - pd.Timedelta(days=1)

        st.markdown("#### ¿Desde qué semana?")
        meses_disp = pd.period_range(primer_inicio, ultimo_dato, freq="M")
        etiquetas = [f"{MESES[p.month - 1].capitalize()} {p.year}" for p in meses_disp]
        if st.session_state.get("mes") not in etiquetas:
            st.session_state["mes"] = etiquetas[list(meses_disp).index(pd.Period(inicio, "M"))]
        mes_sel = st.selectbox("Mes", etiquetas, key="mes", label_visibility="collapsed")
        periodo = meses_disp[etiquetas.index(mes_sel)]

        cab = st.columns(6)
        for c, dname in zip(cab, DIAS):
            c.markdown(f"<div style='text-align:center;font-size:13px;opacity:.7'>{dname}</div>", unsafe_allow_html=True)
        s = domingo_de(periodo.start_time)
        while s <= periodo.end_time:
            elegida = inicio <= s <= fin
            ok = inicio_valido(s, semanas) or elegida
            fila = st.columns(6)
            for j, c in enumerate(fila):
                dia = s + pd.Timedelta(days=j)
                etiqueta = f"{dia.day}" if dia.month == periodo.month else f"·{dia.day}·"
                c.button(etiqueta, key=f"d{dia:%Y%m%d}", disabled=not ok, type="primary" if elegida else "secondary", width="stretch",
                         on_click=lambda s=s: st.session_state.update(inicio=s))
            s += pd.Timedelta(weeks=1)
        st.caption("Toca cualquier día para elegir su semana. Las semanas grises no tienen datos suficientes para ese horizonte.")

    hist = diaria[diaria["fecha"] < inicio]
    periodo_df = diaria[(diaria["fecha"] >= inicio) & (diaria["fecha"] <= fin)].copy()
    with st.spinner("La red está calculando…"):
        periodo_df["lstm"] = pronosticar(cargar_modelo(), hist["ventas"].to_numpy(), hist["fecha"], list(periodo_df["fecha"]), cfg)
    total_p, total_r = periodo_df["lstm"].sum(), periodo_df["ventas"].sum()
    dif = 100 * (total_p - total_r) / total_r

    with der:
        texto_h = {"Semana": "la semana", "Mes": "el mes", "Trimestre": "el trimestre"}[horizonte]
        st.markdown(f"#### Pronóstico para {texto_h} del {corta(inicio)} al {corta(fin - pd.Timedelta(days=1))} de {fin.year}")
        c1, c2, c3 = st.columns(3)
        c1.metric("La red pronosticó", libras(total_p))
        c2.metric("Se vendió", libras(total_r))
        c3.metric("La red se quedó corta" if dif < 0 else "La red se pasó", f"{abs(dif):.0f}%")

        # gráfico: diario para una semana, semanal para mes o trimestre
        if horizonte == "Semana":
            antes = hist.tail(18)[["fecha", "ventas"]]
            g_real = pd.concat([antes, periodo_df[["fecha", "ventas"]]])
            g_pron = periodo_df[["fecha", "lstm"]].rename(columns={"lstm": "ventas"})
            eje_x = alt.X("fecha:T", title=None, axis=alt.Axis(labelExpr=ETIQ_FECHA, labelAngle=0, tickCount=8))
        else:
            sem = lambda df, col: df.assign(semana=df["fecha"].apply(domingo_de)).groupby("semana", as_index=False)[col].sum().rename(columns={"semana": "fecha", col: "ventas"})
            g_real = pd.concat([sem(hist[hist["fecha"] >= inicio - pd.Timedelta(weeks=10)], "ventas"), sem(periodo_df, "ventas")])
            g_pron = sem(periodo_df, "lstm")
            eje_x = alt.X("fecha:T", title="Semana que inicia", axis=alt.Axis(labelExpr=ETIQ_FECHA, labelAngle=0))
        g_real["serie"], g_pron["serie"] = "Venta real", "Red LSTM"
        y_max = max(g_real["ventas"].max(), g_pron["ventas"].max()) * 1.15 / 1000
        datos_g = pd.concat([g_real, g_pron]).assign(mil=lambda d: d["ventas"] / 1000)
        escala = alt.Scale(domain=["Venta real", "Red LSTM"], range=[GRIS, AZUL])
        base = alt.Chart(datos_g).encode(
            x=eje_x,
            y=alt.Y("mil:Q", title=None, scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(labelExpr="'£' + datum.value + ' mil'", grid=True, tickCount=5)),
            color=alt.Color("serie:N", scale=escala, legend=None),
        )
        lineas = base.mark_line(strokeWidth=3.5, point=alt.OverlayMarkDef(size=45)).encode(
            tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d %b %Y"), alt.Tooltip("serie:N", title=""), alt.Tooltip("ventas:Q", title="£", format=",.0f")])
        ultimos = datos_g.sort_values("fecha").groupby("serie").tail(1)
        arriba = ultimos.set_index("serie")["mil"]
        ultimos["dy"] = np.where((ultimos["serie"] == "Red LSTM") == (arriba["Red LSTM"] >= arriba["Venta real"]), -10, 10)
        etiquetas_g = alt.layer(*[
            alt.Chart(ultimos[ultimos["dy"] == d]).mark_text(align="left", dx=10, dy=d, fontSize=15, fontWeight="bold").encode(
                x="fecha:T", y="mil:Q", text="serie:N", color=alt.Color("serie:N", scale=escala, legend=None))
            for d in (-10, 10)])
        sombra = alt.Chart(pd.DataFrame({"a": [inicio - pd.Timedelta(hours=12)], "b": [g_real["fecha"].max() + pd.Timedelta(hours=12)]})).mark_rect(opacity=0.08, color=AZUL).encode(x="a:T", x2="b:T")
        rot = alt.Chart(pd.DataFrame({"a": [inicio], "t": ["Periodo pronosticado"]})).mark_text(align="left", dx=4, dy=-8, fontSize=12, opacity=0.7).encode(
            x="a:T", y=alt.value(12), text="t:N")
        st.altair_chart((sombra + lineas + etiquetas_g + rot).properties(height=360, padding={"right": 90}), width="stretch")

        ref = hz.get(horizonte)
        if ref:
            st.caption(f"Referencia ({ref['n']} periodos de prueba): error típico de la red al planear {texto_h}: **{ref['lstm']:.0f}%** · repitiendo {texto_h} anterior: **{ref['anterior']:.0f}%**.")
        if horizonte == "Trimestre":
            st.warning("Horizonte largo: la red se entrenó para pronosticar el día siguiente y en un trimestre acumula error. En la temporada navideña se quedó corta entre 25% y 40%.")
        if inicio < inicio_prueba:
            st.info("La red vio estas semanas al entrenar, así que la comparación es optimista. El periodo de prueba empieza el 2 de octubre de 2011.")

        with st.expander("Ver detalle por semana"):
            det = periodo_df.assign(Semana=periodo_df["fecha"].apply(domingo_de)).groupby("Semana", as_index=False)[["lstm", "ventas"]].sum()
            det["Diferencia"] = (100 * (det["lstm"] - det["ventas"]) / det["ventas"]).map(lambda v: f"{v:+.0f}%")
            det["Semana"] = det["Semana"].map(lambda f: f"{corta(f)} – {corta(f + pd.Timedelta(days=5))}")
            det["Red LSTM"] = det["lstm"].map(libras)
            det["Venta real"] = det["ventas"].map(libras)
            st.dataframe(det[["Semana", "Red LSTM", "Venta real", "Diferencia"]], hide_index=True, width="stretch")

# ---------- pestaña 2: evaluación ----------
with tab2:
    st.subheader("Prueba: últimos 60 días con venta (oct – dic 2011)")
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
        largo = prueba.melt(id_vars="fecha", value_vars=["real", "LSTM"], var_name="serie", value_name="ventas")
        largo["serie"] = largo["serie"].replace({"real": "Venta real", "LSTM": "Red LSTM"})
        st.altair_chart(alt.Chart(largo).mark_line(strokeWidth=3).encode(
            x=alt.X("fecha:T", title=None), y=alt.Y("ventas:Q", title="Ventas (£)"),
            color=alt.Color("serie:N", scale=alt.Scale(domain=["Venta real", "Red LSTM"], range=[GRIS, AZUL]), legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("fecha:T", format="%d %b"), "serie:N", alt.Tooltip("ventas:Q", format=",.0f")],
        ).properties(height=300), width="stretch")
        curvas = pd.DataFrame({"época": range(1, len(cfg["curvas"]["loss"]) + 1), "Entrenamiento": cfg["curvas"]["loss"], "Validación": cfg["curvas"]["val_loss"]}).melt("época", var_name="serie", value_name="pérdida")
        st.markdown("**Curvas de aprendizaje (pérdida Huber)**")
        st.altair_chart(alt.Chart(curvas).mark_line().encode(
            x="época:Q", y=alt.Y("pérdida:Q", title="Pérdida"),
            color=alt.Color("serie:N", scale=alt.Scale(range=[AZUL, "#e3a33c"]), legend=alt.Legend(orient="top", title=None)),
        ).properties(height=200), width="stretch")

# ---------- pestaña 3: SQL ----------
SQL_CALIDAD = """-- 1. ¿Qué hay que limpiar? Cuenta los registros problemáticos
SELECT
    SUM(InvoiceNo LIKE 'C%')  AS facturas_canceladas,  -- empiezan por C
    SUM(Quantity  <= 0)       AS cantidades_negativas,
    SUM(UnitPrice <= 0)       AS precios_en_cero
FROM transactions;"""

SQL_DIARIA = """-- 2. Venta diaria: la serie que aprende la red
SELECT
    DATE(InvoiceDate)                 AS fecha,
    SUM(Quantity * UnitPrice)         AS ventas,    -- £ del día
    COUNT(DISTINCT InvoiceNo)         AS pedidos
FROM transactions
WHERE Quantity  > 0                   -- sin devoluciones
  AND UnitPrice > 0                   -- sin precios en cero
  AND InvoiceNo NOT LIKE 'C%'         -- sin facturas canceladas
GROUP BY DATE(InvoiceDate)
ORDER BY fecha;"""

EJEMPLOS = {
    "Ventas por mes": """SELECT
    strftime('%Y-%m', fecha)  AS mes,
    SUM(ventas)               AS ventas,
    SUM(pedidos)              AS pedidos
FROM ventas_diarias
GROUP BY mes
ORDER BY mes;""",
    "Venta promedio por día de la semana": """SELECT
    CASE strftime('%w', fecha)
        WHEN '1' THEN 'Lunes'  WHEN '2' THEN 'Martes'  WHEN '3' THEN 'Miércoles'
        WHEN '4' THEN 'Jueves' WHEN '5' THEN 'Viernes' WHEN '0' THEN 'Domingo'
    END                       AS dia,
    COUNT(*)                  AS dias_con_venta,
    AVG(ventas)               AS venta_promedio
FROM ventas_diarias
GROUP BY strftime('%w', fecha)
ORDER BY (strftime('%w', fecha) + 6) % 7;   -- lunes primero""",
    "Los 10 días de mayor venta": """SELECT fecha, ventas, pedidos
FROM ventas_diarias
ORDER BY ventas DESC
LIMIT 10;""",
}


def mostrar_tabla(df):
    vista = df.copy()
    if "mes" in vista.columns:
        vista["mes"] = vista["mes"].map(lambda t: f"{MESES[int(t[5:7]) - 1].capitalize()} {t[:4]}")
    for col in vista.columns:
        if "venta" in col and pd.api.types.is_numeric_dtype(vista[col]):
            vista[col] = vista[col].map(libras)
        elif pd.api.types.is_numeric_dtype(vista[col]):
            vista[col] = vista[col].map(miles)
    vista.columns = [c.replace("_", " ").capitalize() for c in vista.columns]
    st.dataframe(vista, hide_index=True, width="stretch")


with tab3:
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Cómo se armaron los datos")
        st.markdown("El notebook carga las **541.909 líneas de factura** en una base **SQLite** (tabla `transactions`). "
                    "Estas dos consultas limpian los datos y arman la venta diaria que aprende la red.")
        st.code(SQL_CALIDAD, language="sql")
        st.code(SQL_DIARIA, language="sql")
        st.caption("Resultado: 305 días con venta, guardados en la tabla `ventas_diarias`.")
    with b:
        st.subheader("Consulta la venta diaria")
        st.markdown("Base SQLite real con la tabla `ventas_diarias` (fecha, ventas, pedidos). Solo lectura.")
        elegido = st.selectbox("Consulta de ejemplo", list(EJEMPLOS))
        consulta = st.text_area("SQL", EJEMPLOS[elegido], height=220, key=f"sql_{elegido}")
        if st.button("▶ Ejecutar", type="primary"):
            if not consulta.strip().lower().startswith("select"):
                st.error("Solo se permiten consultas SELECT.")
            else:
                con = sqlite3.connect(":memory:")
                diaria.assign(fecha=diaria["fecha"].dt.strftime("%Y-%m-%d")).to_sql("ventas_diarias", con, index=False)
                try:
                    mostrar_tabla(pd.read_sql(consulta, con))
                except Exception as e:
                    st.error(f"Error en la consulta: {e}")
                finally:
                    con.close()
        st.caption("SQLite guarda las fechas como texto (2011-11-27). `strftime` extrae el mes o el día de la semana. El formato de moneda se aplica al mostrar la tabla.")

# ---------- pestaña 4: cómo funciona ----------
with tab4:
    st.subheader("Del dato a la decisión")
    st.markdown(f"""
1. **SQL (SQLite):** 541.909 líneas de factura → limpieza (sin cancelaciones ni precios en cero) → **venta diaria** (305 días).
2. **Preparación:** a cada ventana de {cfg['ventana']} días se le resta su promedio. La red aprende la *forma* de las ventas (el ritmo de la semana) y no el nivel, por eso se adapta a la temporada navideña.
3. **Red LSTM:** lee {cfg['ventana']} días en orden y pronostica el día siguiente. Para una semana o un mes, cada pronóstico entra a la ventana del siguiente día.
4. **Evaluación:** octubre a diciembre de 2011, periodo que la red nunca vio, contra los métodos simples de un planificador.

| Capa | Configuración |
|---|---|
| Entrada | {cfg['ventana']} días × 7 variables (venta + día de la semana) |
| LSTM | 32 unidades |
| Dropout | 0,2 |
| Densa | 16, ReLU |
| Salida | 1 (venta del día siguiente) |

Adam (0,001) · pérdida Huber · parada temprana · {cfg['epocas']} épocas. Se comparó con una GRU y con un MLP.

**Límite:** la red sirve para el corto plazo (días y semanas). En un mes empata con repetir el mes anterior. Para planear meses o trimestres haría falta entrenar con datos mensuales de varios años.
""")
    st.caption("Código y notebook: github.com/gigasanti/aca1-grupo1-cun (carpeta aa2/)")
