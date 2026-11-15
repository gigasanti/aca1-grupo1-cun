"""ACA 2 · Aprendizaje Automático · Grupo 1 CUN
Pronóstico de ventas (UCI Online Retail) con una red LSTM entrenada en el notebook ACA2_AA_LSTM_Ventas_Grupo1.ipynb.
"""
import json
import os
import re
import sqlite3
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
AZUL, GRIS, GRIS_OSC = "#3b7dd8", "#9aa3ad", "#6b737c"
TEXTO = "#8c96a3"   # gris medio: legible en modo claro y oscuro
DIAS = ["Dom", "Lun", "Mar", "Mié", "Jue", "Vie"]          # la tienda no vende los sábados
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
HORIZONTES = ["Día", "Semana"]
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
def cargar_modelo(huella=None):
    import tensorflow as tf
    return tf.keras.models.load_model(BASE / "modelo_lstm.keras")


@st.cache_data
def cargar_datos(huella):
    """`huella` cambia cuando cambian los archivos, así Streamlit no reutiliza datos viejos en memoria."""
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


HUELLA = tuple((BASE / f).stat().st_mtime_ns for f in ("config.json", "ventas_diarias.csv", "prueba_pronosticos.csv"))
cfg, diaria, prueba = cargar_datos(HUELLA)
m = cfg["metricas"]
hz = cfg["horizontes"]
inicio_prueba = diaria["fecha"].iloc[-cfg["dias_prueba"]]
ultimo_dato = diaria["fecha"].iloc[-1]
primer_inicio = domingo_de(diaria["fecha"].iloc[cfg["ventana"]]) + pd.Timedelta(weeks=1)
base_mae = min(v["MAE (£)"] for k, v in m.items() if k not in ("LSTM", "GRU"))


fechas_venta = set(diaria["fecha"])
primer_dia = diaria["fecha"].iloc[cfg["ventana"]]


def semana_valida(s):
    # solo semanas del periodo de evaluación: ventas que la red nunca vio al entrenar
    return inicio_prueba <= s and s + pd.Timedelta(days=6) <= ultimo_dato + pd.Timedelta(days=1)


def dia_valido(f):
    return f in fechas_venta and f >= inicio_prueba


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
    st.session_state.setdefault("inicio", pd.Timestamp("2011-11-20"))
    st.session_state.setdefault("dia", pd.Timestamp("2011-11-21"))
    izq, der = st.columns([1, 2.3], gap="large")

    with izq:
        st.markdown("#### ¿Qué quieres pronosticar?")
        horizonte = st.segmented_control("Horizonte", HORIZONTES, default="Semana", key="horizonte", label_visibility="collapsed") or "Semana"
        if horizonte == "Semana":
            inicio = st.session_state["inicio"]
            fin = inicio + pd.Timedelta(days=6)
            ancla = inicio
        else:
            inicio = fin = st.session_state["dia"]
            ancla = inicio

        st.markdown("#### " + ("¿Qué semana?" if horizonte == "Semana" else "¿Qué día?"))
        meses_disp = pd.period_range(inicio_prueba, ultimo_dato, freq="M")
        etiquetas = [f"{MESES[p.month - 1].capitalize()} {p.year}" for p in meses_disp]
        if st.session_state.get("mes") not in etiquetas:
            st.session_state["mes"] = etiquetas[list(meses_disp).index(pd.Period(ancla, "M"))]
        mes_sel = st.selectbox("Mes", etiquetas, key="mes", label_visibility="collapsed")
        periodo = meses_disp[etiquetas.index(mes_sel)]

        cab = st.columns(6)
        for c, dname in zip(cab, DIAS):
            c.markdown(f"<div style='text-align:center;font-size:13px;opacity:.7'>{dname}</div>", unsafe_allow_html=True)
        s = domingo_de(periodo.start_time)
        while s <= periodo.end_time:
            fila = st.columns(6)
            for j, c in enumerate(fila):
                dia = s + pd.Timedelta(days=j)
                etiqueta = f"{dia.day}" if dia.month == periodo.month else f"·{dia.day}·"
                if horizonte == "Semana":
                    elegido, ok = inicio <= dia <= fin, semana_valida(s)
                    accion = lambda s=s: st.session_state.update(inicio=s)
                else:
                    elegido, ok = dia == inicio, dia_valido(dia)
                    accion = lambda dia=dia: st.session_state.update(dia=dia)
                c.button(etiqueta, key=f"d{dia:%Y%m%d}", disabled=not (ok or elegido), type="primary" if elegido else "secondary",
                         width="stretch", on_click=accion)
            s += pd.Timedelta(weeks=1)
        st.caption("Solo se puede elegir de octubre a diciembre de 2011: la red se entrenó con los meses anteriores y nunca vio estas ventas. Así se compara su pronóstico con lo que de verdad se vendió.")

    hist = diaria[diaria["fecha"] < inicio]
    periodo_df = diaria[(diaria["fecha"] >= inicio) & (diaria["fecha"] <= fin)].copy()
    with st.spinner("La red está calculando…"):
        periodo_df["lstm"] = pronosticar(cargar_modelo((BASE / "modelo_lstm.keras").stat().st_mtime_ns), hist["ventas"].to_numpy(), hist["fecha"], list(periodo_df["fecha"]), cfg)
    total_p, total_r = periodo_df["lstm"].sum(), periodo_df["ventas"].sum()
    dif = 100 * (total_p - total_r) / total_r

    with der:
        if horizonte == "Semana":
            st.markdown(f"#### Pronóstico para la semana del {corta(inicio)} al {corta(fin - pd.Timedelta(days=1))} de {fin.year}")
        else:
            dias_largos = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
            st.markdown(f"#### Pronóstico para el {dias_largos[inicio.dayofweek]} {corta(inicio)} de {inicio.year}")
        c1, c2, c3 = st.columns(3)
        c1.metric("La red pronosticó", libras(total_p))
        c2.metric("Se vendió", libras(total_r))
        c3.metric("La red se quedó corta" if dif < 0 else "La red se pasó", f"{abs(dif):.0f}%")

        # lectura ejecutiva: qué habría pasado si se planeaba con este número, y contra la regla simple
        if horizonte == "Semana":
            regla = diaria[(diaria["fecha"] >= inicio - pd.Timedelta(days=7)) & (diaria["fecha"] < inicio)]["ventas"].sum()
            regla_txt = "repetir la semana anterior"
        else:
            previo = diaria.loc[diaria["fecha"] == inicio - pd.Timedelta(days=7), "ventas"]
            regla = float(previo.iloc[0]) if len(previo) else np.nan
            regla_txt = "repetir el mismo día de la semana pasada"
        brecha = total_p - total_r
        if brecha >= 0:
            efecto = (f"Si se compraba inventario y se programaba personal con este número, se planeaba para **{libras(brecha)} más** de lo que se vendió: "
                      "sobra producto en bodega y horas de trabajo. No es dinero perdido, es costo de tener de más.")
        else:
            efecto = (f"Si se planeaba con este número, se preparaba para **{libras(-brecha)} menos** de lo que se vendió: "
                      "riesgo de agotados y de perder parte de esa venta.")
        if np.isnan(regla):
            comparacion = ""
        else:
            dif_regla = 100 * (regla - total_r) / total_r
            gana = abs(dif) < abs(dif_regla)
            comparacion = (f" La regla de {regla_txt} habría dicho **{libras(regla)}** ({dif_regla:+.0f}%). "
                           + ("**En este caso la red estuvo más cerca.**" if gana else "**En este caso la regla simple estuvo más cerca:** pasa en algunos periodos, la red gana en el promedio."))
        st.markdown(f"<div style='border-left:5px solid {AZUL};padding:10px 16px;border-radius:6px;background:rgba(59,125,216,.08);margin:4px 0 8px'>"
                    f"<b>Lectura ejecutiva.</b> {re.sub(r'[*][*](.+?)[*][*]', r'<b>\1</b>', efecto + comparacion)}</div>", unsafe_allow_html=True)

        # gráfico: diario para una semana, semanal para mes o trimestre
        if True:
            antes = hist.tail(18 if horizonte == "Semana" else 24)[["fecha", "ventas"]]
            g_real = pd.concat([antes, periodo_df[["fecha", "ventas"]]])
            g_pron = periodo_df[["fecha", "lstm"]].rename(columns={"lstm": "ventas"})
            eje_x = alt.X("fecha:T", title=None, axis=alt.Axis(labelExpr=ETIQ_FECHA, labelAngle=0, tickCount=8))
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
        sombra = alt.Chart(pd.DataFrame({"a": [inicio - pd.Timedelta(hours=14)], "b": [g_real["fecha"].max() + pd.Timedelta(hours=14)]})).mark_rect(opacity=0.15, color=AZUL).encode(x="a:T", x2="b:T")
        rot = alt.Chart(pd.DataFrame({"a": [inicio - pd.Timedelta(hours=14)], "t": ["Pronóstico ▸"]})).mark_text(align="right", dx=-6, fontSize=13, fontWeight="bold", color=AZUL).encode(
            x="a:T", y=alt.value(14), text="t:N")
        st.altair_chart((sombra + lineas + etiquetas_g + rot).properties(height=360, padding={"right": 90}), width="stretch")

        if horizonte == "Semana":
            st.caption(f"Referencia ({hz['Semana']['n']} semanas de prueba): error típico de la red al pronosticar una semana **{hz['Semana']['lstm']:.0f}%** · repitiendo la semana anterior **{hz['Semana']['anterior']:.0f}%**.")
        else:
            st.caption(f"Referencia (60 días de prueba): la red se equivoca en promedio **{libras_mil(m['LSTM']['MAE (£)'])}** por día ({m['LSTM']['MAPE (%)']:.0f}% de la venta del día) · repitiendo el mismo día de la semana pasada **{libras_mil(m['Ingenuo estacional (semana pasada)']['MAE (£)'])}**.")

        if horizonte == "Semana":
          with st.expander("Ver detalle por día"):
            det = periodo_df.copy()
            dias_largos = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            det["Día"] = det["fecha"].map(lambda f: f"{dias_largos[f.dayofweek]} {corta(f)}")
            det["Red LSTM"] = det["lstm"].map(libras)
            det["Venta real"] = det["ventas"].map(libras)
            det["Diferencia"] = (100 * (det["lstm"] - det["ventas"]) / det["ventas"]).map(lambda v: f"{v:+.0f}%")
            st.dataframe(det[["Día", "Red LSTM", "Venta real", "Diferencia"]], hide_index=True, width="stretch")

# ---------- pestaña 2: evaluación ----------
with tab2:
    st.markdown("#### Examen de la red: 60 días que nunca vio (2 oct – 9 dic 2011)")
    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("**1. ¿Cuánto se equivoca cada método por día?**")
        nombres = {"LSTM": "Red LSTM", "GRU": "Red GRU (comparación)", "Media móvil 6 días": "Promedio últimos 6 días",
                   "Ingenuo estacional (semana pasada)": "Repetir semana pasada", "Ingenuo (día anterior)": "Repetir el día anterior"}
        err = pd.DataFrame([{"Método": nombres[k], "mil": v["MAE (£)"] / 1000, "pct": v["MAPE (%)"]} for k, v in m.items()]).sort_values("mil")
        err["texto"] = err.apply(lambda r: f"£{r['mil']:.1f} mil · {r['pct']:.0f}%".replace(".", ","), axis=1)
        err["color"] = np.where(err["Método"] == "Red LSTM", AZUL, GRIS)
        barras = alt.Chart(err).encode(y=alt.Y("Método:N", sort=list(err["Método"]), title=None, axis=alt.Axis(labelFontSize=13, labelLimit=220)),
                                       x=alt.X("mil:Q", title=None, axis=None, scale=alt.Scale(domain=[0, err["mil"].max() * 1.45])))
        st.altair_chart((barras.mark_bar(cornerRadiusEnd=5, height=24).encode(color=alt.Color("color:N", scale=None))
                         + barras.mark_text(align="left", dx=6, fontSize=13, color=TEXTO).encode(text="texto:N")).properties(height=230), width="stretch")
        st.caption("Libras de diferencia promedio entre el pronóstico y la venta real de cada día (MAE), y ese error como % de la venta del día (MAPE). Menos es mejor.")

        st.markdown("**2. ¿Acierta si mañana se vende más o menos que hoy?**")
        cm = np.array(cfg["direccion"]["matriz"])
        aciertos, total = int(cm[0, 0] + cm[1, 1]), int(cm.sum())
        d1, d2, d3 = st.columns(3)
        d1.metric("Acertó", f"{aciertos} de {total} días")
        d2.metric("Días que subieron", f"{cm[1, 1]} de {cm[1].sum()}", help="De los días en que la venta subió frente al día anterior, cuántos anticipó la red.")
        d3.metric("Días que bajaron", f"{cm[0, 0]} de {cm[0].sum()}", help="De los días en que la venta bajó frente al día anterior, cuántos anticipó la red.")
        st.caption(f"Sirve para decidir si mañana se refuerza o se reduce el turno. Nota técnica: F1 = {cfg['direccion']['f1_macro']:.2f} (promedia el acierto en días que suben y en días que bajan).".replace("0.", "0,"))

    with b:
        st.markdown("**3. Pronóstico frente a la venta real, día por día**")
        largo = prueba.melt(id_vars="fecha", value_vars=["real", "LSTM"], var_name="serie", value_name="ventas")
        largo["serie"] = largo["serie"].replace({"real": "Venta real", "LSTM": "Red LSTM"})
        largo["mil"] = largo["ventas"] / 1000
        escala2 = alt.Scale(domain=["Venta real", "Red LSTM"], range=[GRIS, AZUL])
        meses_ticks = [pd.Timestamp("2011-10-04"), pd.Timestamp("2011-11-01"), pd.Timestamp("2011-12-01")]
        lineas2 = alt.Chart(largo).mark_line(strokeWidth=3).encode(
            x=alt.X("fecha:T", title=None, axis=alt.Axis(values=[t.isoformat() for t in meses_ticks], labelExpr="['ene','feb','mar','abr','may','jun','jul','ago','sep','Octubre','Noviembre','Diciembre'][month(datum.value)]", labelFontSize=13, grid=False)),
            y=alt.Y("mil:Q", title=None, scale=alt.Scale(domain=[0, 130], clamp=True), axis=alt.Axis(labelExpr="'£' + datum.value + ' mil'", tickCount=4)),
            color=alt.Color("serie:N", scale=escala2, legend=None),
            tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"), alt.Tooltip("serie:N", title=""), alt.Tooltip("ventas:Q", title="£", format=",.0f")])
        fin_g = largo.sort_values("fecha").groupby("serie").tail(1)
        fin_g["mil"] = fin_g["mil"].clip(upper=125)
        fin_g["dy"] = np.where(fin_g["serie"] == "Venta real", -10, 10)
        etq2 = alt.layer(*[alt.Chart(fin_g[fin_g["dy"] == d]).mark_text(align="left", dx=8, dy=d, fontSize=14, fontWeight="bold").encode(
            x="fecha:T", y="mil:Q", text="serie:N", color=alt.Color("serie:N", scale=escala2, legend=None)) for d in (-10, 10)])
        st.altair_chart((lineas2 + etq2).properties(height=260, padding={"right": 95}), width="stretch")
        st.caption("La red sigue el ritmo semanal. Los picos que no alcanza son pedidos excepcionales (14 nov y 9 dic) que ninguna venta pasada anunciaba.")

        st.markdown("**4. ¿La red aprendió o memorizó?**")
        curvas = pd.DataFrame({"Vuelta de entrenamiento": range(1, len(cfg["curvas"]["loss"]) + 1),
                               "Error con datos de entrenamiento": cfg["curvas"]["loss"],
                               "Error con datos que no usó para aprender": cfg["curvas"]["val_loss"]}).melt("Vuelta de entrenamiento", var_name="serie", value_name="error")
        st.altair_chart(alt.Chart(curvas).mark_line(strokeWidth=3).encode(
            x=alt.X("Vuelta de entrenamiento:Q", axis=alt.Axis(tickCount=5, grid=False)),
            y=alt.Y("error:Q", title=None, scale=alt.Scale(domain=[curvas["error"].min() * 0.9, curvas["error"].max() * 1.05]), axis=alt.Axis(labels=False, ticks=False, grid=False)),
            color=alt.Color("serie:N", scale=alt.Scale(range=[GRIS, AZUL]), legend=alt.Legend(orient="top", title=None, labelFontSize=12, labelLimit=320)),
        ).properties(height=230), width="stretch")
        st.caption("Las dos curvas bajan y se estabilizan juntas: la red aprendió un patrón que también funciona con datos nuevos. Si la azul subiera mientras la gris baja, estaría memorizando.")

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
3. **Red LSTM:** lee {cfg['ventana']} días en orden y pronostica el día siguiente. Para una semana, cada pronóstico entra a la ventana del día siguiente.
4. **Evaluación:** octubre a diciembre de 2011, periodo que la red nunca vio, contra los métodos simples de un planificador.

| Capa | Configuración |
|---|---|
| Entrada | {cfg['ventana']} días × 7 variables (venta + día de la semana) |
| LSTM | 32 unidades |
| Dropout | 0,2 |
| Densa | 16, ReLU |
| Salida | 1 (venta del día siguiente) |

Adam (0,001) · pérdida Huber · parada temprana · {cfg['epocas']} épocas. Se comparó con una GRU y con un MLP.

**Límite:** la red sirve para el corto plazo (días y semanas). Se probó a un mes y empató con repetir el mes anterior, por eso la app no lo ofrece. Para planear meses haría falta entrenar con datos mensuales de varios años.
""")
    st.caption("Código y notebook: github.com/gigasanti/aca1-grupo1-cun (carpeta aa2/)")
