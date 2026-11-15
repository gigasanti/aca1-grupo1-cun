# ACA 2 · Aprendizaje Automático · Pronóstico de ventas con LSTM

Pilar **Ventas** · Dataset **UCI Online Retail** (Chen, 2015) · Docente: Dr. Hamilton Rivera Flor

Red neuronal recurrente **LSTM** que pronostica la venta diaria total de la tienda y se compara contra tres métodos simples (día anterior, semana pasada, media móvil).

| Resultado (últimos 60 días, oct – dic 2011) | LSTM | Mejor método simple |
|---|---|---|
| MAE diario | £14.168 | £16.512 (media móvil) |
| MAPE diario | 24,6 % | 32,1 % (semana pasada) |
| F1 dirección del día siguiente (sube/baja) | 0,78 | 0,73 |
| Error típico al planear una semana | 17,7 % | 21,3 % (repetir la semana anterior) |
| Error típico al planear un mes | 18,8 % | 18,6 % (repetir el mes anterior) |

## Contenido

- `ACA2_AA_LSTM_Ventas_Grupo1.ipynb`: notebook completo (SQLite, EDA, preparación, LSTM y GRU, evaluación, exportación).
- `app/`: aplicación Streamlit (`app.py`, `modelo_lstm.keras`, `config.json`, datos diarios y pronósticos de prueba).

## Reproducir

1. Descargar `Online Retail.xlsx` de https://archive.ics.uci.edu/dataset/352/online+retail y dejarlo en `aa2/online+retail/` (el notebook lo descarga solo si no lo encuentra).
2. `pip install -r app/requirements.txt matplotlib scikit-learn openpyxl`
3. Ejecutar el notebook de principio a fin. Escribe el modelo y los datos en `app/`.

## Ejecutar la app

```
cd aa2/app
pip install -r requirements.txt
streamlit run app.py
```

En Streamlit Community Cloud: archivo principal `aa2/app/app.py`.
