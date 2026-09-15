"""Interfaz del sistema de detección de lavado de dinero en remesas.

Permite seleccionar un remitente del conjunto de prueba, ver su secuencia de
transacciones, los puntajes de ambas etapas, un mapa de calor de que
transacciones activaron la alerta, y una explicación escrita de la decisión.

La inferencia es real y corre sobre el modelo ONNX exportado por el segundo
cuaderno, no sobre resultados precalculados.

Uso:
    streamlit run src/inference/deployment/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[3]
sys.path.append(str(RAIZ / "src"))

from inference.utils.explicacion import Resumen, contribucion, explicar  # noqa: E402
from inference.utils.vista import contexto_remitente  # noqa: E402

DATOS = Path(__file__).resolve().parent / "datos"
MODELO = RAIZ / "results" / "models" / "sistema.onnx"

COLOR_ALERTA = "#C44E52"
COLOR_NORMAL = "#4C72B0"

COLUMNAS_TABLA = [
    "transaccion", "fecha", "monto_usd", "formato", "moneda", "destino",
    "destino_nuevo", "cruce_moneda", "cerca_umbral", "contribucion", "es_lavado",
]


@st.cache_resource(show_spinner=False)
def cargar_sesion() -> onnxruntime.InferenceSession:
    """Abre el modelo ONNX una sola vez por proceso."""
    return onnxruntime.InferenceSession(str(MODELO), providers=["CPUExecutionProvider"])


@st.cache_data(show_spinner=False)
def cargar_datos() -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Lee la muestra de remitentes, su contexto y los metadatos."""
    with np.load(DATOS / "muestra.npz", allow_pickle=False) as bloque:
        muestra = {clave: bloque[clave] for clave in bloque.files}
    contexto = pd.read_parquet(DATOS / "contexto.parquet")
    meta = json.loads((DATOS / "meta.json").read_text(encoding="utf-8"))
    return muestra, contexto, meta


@st.cache_data(show_spinner=False)
def puntuar_todos(_sesion: onnxruntime.InferenceSession, X: np.ndarray, mascara: np.ndarray) -> dict:
    """Ejecuta el modelo sobre la muestra completa y devuelve todas las salidas."""
    salidas = _sesion.run(None, {"x": X.astype(np.float32), "mascara": mascara.astype(np.float32)})
    nombres = ["probabilidad", "probabilidad_etapa_b", "error", "error_por_paso", "atencion"]
    return dict(zip(nombres, salidas))


def barra_de_puntaje(valor: float, umbral: float, titulo: str, formato: str) -> go.Figure:
    """Indicador con el valor obtenido y una línea en el umbral operativo."""
    figura = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        title={"text": titulo, "font": {"size": 14}},
        number={"valueformat": formato},
        gauge={
            "axis": {"range": [0, max(valor, umbral) * 1.25]},
            "bar": {"color": COLOR_ALERTA if valor >= umbral else COLOR_NORMAL},
            "threshold": {"line": {"color": "black", "width": 3}, "value": umbral},
        },
    ))
    figura.update_layout(height=200, margin={"l": 20, "r": 20, "t": 40, "b": 10})
    return figura


def mapa_de_calor(pesos: pd.DataFrame) -> go.Figure:
    """Mapa de calor de la contribución de cada transacción a la alerta."""
    normalizar = lambda v: (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else v * 0
    figura = go.Figure(go.Heatmap(
        z=[normalizar(pesos["atencion"].to_numpy()), normalizar(pesos["anomalia"].to_numpy())],
        x=[str(paso + 1) for paso in pesos["paso"]],
        y=["atención (etapa B)", "anomalía (etapa A)"],
        colorscale="Reds",
        zmin=0, zmax=1,
        colorbar={"title": "contribución"},
        hovertemplate="transacción %{x}<br>%{y}<br>contribución %{z:.2f}<extra></extra>",
    ))
    figura.update_layout(
        height=220, margin={"l": 10, "r": 10, "t": 30, "b": 40},
        xaxis_title="transacción de la secuencia",
    )
    return figura


def main() -> None:
    """Arma la página completa."""
    st.set_page_config(page_title="Detección de Lavado en Remesas", layout="wide")

    if not MODELO.exists():
        st.error(
            f"No se encontró el modelo en {MODELO}. Ejecute el cuaderno "
            "`02_deteccion_dos_etapas.ipynb` para generarlo."
        )
        return

    muestra, contexto, meta = cargar_datos()
    resultados = puntuar_todos(cargar_sesion(), muestra["X"], muestra["mascara"])
    umbrales = meta["umbrales"]

    st.title("Detección de lavado de dinero en secuencias de remesas")
    st.caption(
        f"Sistema de dos etapas sobre una muestra de {meta['remitentes']} remitentes del conjunto "
        f"de prueba. La muestra conserva los {meta['positivos']} casos de lavado y completa con "
        "negativos tomados al azar, asi que sobrerrepresenta a la clase positiva a propósito. "
        "Los conteos de esta página describen la muestra, no la tasa de operación real, que "
        f"corresponde a un presupuesto de revisión del {umbrales['presupuesto']:.0%} del volumen."
    )

    alerta = resultados["probabilidad"] >= umbrales["alerta"]
    etiquetas = pd.DataFrame({
        "remitente": muestra["identificador"],
        "etiqueta": muestra["y"],
        "probabilidad": resultados["probabilidad"],
        "alerta": alerta,
    })

    with st.sidebar:
        st.header("Selección de remitente")
        filtro = st.radio(
            "Mostrar",
            ["Todos", "Solo los que generan alerta", "Solo los etiquetados como lavado",
             "Solo los errores del sistema"],
        )
        visibles = etiquetas
        if filtro == "Solo los que generan alerta":
            visibles = etiquetas[etiquetas["alerta"]]
        elif filtro == "Solo los etiquetados como lavado":
            visibles = etiquetas[etiquetas["etiqueta"] == 1]
        elif filtro == "Solo los errores del sistema":
            visibles = etiquetas[etiquetas["alerta"] != (etiquetas["etiqueta"] == 1)]

        visibles = visibles.sort_values("probabilidad", ascending=False)
        if visibles.empty:
            st.warning("Ningún remitente cumple ese filtro.")
            return

        opciones = visibles.index.to_list()
        elegido = st.selectbox(
            f"Remitente  ({len(opciones)} disponibles)",
            opciones,
            format_func=lambda i: (
                f"{etiquetas.loc[i, 'remitente']}  "
                f"[{'lavado' if etiquetas.loc[i, 'etiqueta'] else 'normal'}]  "
                f"{etiquetas.loc[i, 'probabilidad']:.0%}"
            ),
        )
        st.divider()
        st.metric("Con alerta en la muestra", f"{int(alerta.sum())} de {len(alerta)}")
        st.metric("Positivos detectados",
                  f"{int((alerta & (muestra['y'] == 1)).sum())} de {int((muestra['y'] == 1).sum())}")
        st.caption(
            "La muestra sobrerrepresenta los casos de lavado, así que estas proporciones no son "
            "las que vería el sistema en operación."
        )

    pasos = int(muestra["longitud"][elegido])
    vista = contexto_remitente(
        contexto, elegido, muestra["X"][elegido], muestra["mascara"][elegido],
        meta["variables"], meta["catalogos"],
    )
    pesos = contribucion(
        resultados["atencion"][elegido], resultados["error_por_paso"][elegido],
        muestra["mascara"][elegido],
    )
    resumen = Resumen(
        identificador=str(muestra["identificador"][elegido]),
        probabilidad=float(resultados["probabilidad"][elegido]),
        error=float(resultados["error"][elegido]),
        umbral_probabilidad=float(umbrales["alerta"]),
        umbral_error=float(umbrales["anomalia"]),
    )

    encabezado = st.columns([2, 1, 1, 1])
    encabezado[0].subheader(resumen.identificador)
    encabezado[0].write(
        f"**{'ALERTA' if resumen.alerta else 'sin alerta'}**  ·  "
        f"{pasos} transacciones  ·  "
        f"etiqueta real **{'lavado' if muestra['y'][elegido] else 'normal'}**"
    )
    encabezado[1].plotly_chart(
        barra_de_puntaje(resumen.probabilidad, umbrales["alerta"],
                         "Probabilidad combinada", ".1%"),
        width="stretch",
    )
    encabezado[2].plotly_chart(
        barra_de_puntaje(float(resultados["probabilidad_etapa_b"][elegido]), umbrales["alerta"],
                         "Etapa B, clasificador", ".1%"),
        width="stretch",
    )
    encabezado[3].plotly_chart(
        barra_de_puntaje(resumen.error, umbrales["anomalia"],
                         "Etapa A, anomalía", ".3f"),
        width="stretch",
    )

    st.subheader("¿Qué transacciones activaron la alerta?")
    st.plotly_chart(mapa_de_calor(pesos), width="stretch")

    st.subheader("Explicación")
    st.info(explicar(resumen, vista, pesos))

    st.subheader("Secuencia de transacciones")
    tabla = vista.join(pesos.set_index("paso")[["contribucion"]], on="paso")
    tabla["transaccion"] = tabla["paso"] + 1
    for indicador in ("destino_nuevo", "cruce_moneda", "cerca_umbral", "es_lavado"):
        tabla[indicador] = tabla[indicador].astype(bool)

    st.dataframe(
        tabla[COLUMNAS_TABLA],
        width="stretch",
        hide_index=True,
        column_config={
            "transaccion": st.column_config.NumberColumn("N", width="small"),
            "fecha": st.column_config.DatetimeColumn("Fecha", format="DD/MM/YYYY HH:mm"),
            "monto_usd": st.column_config.NumberColumn("Monto USD", format="$%.2f"),
            "formato": st.column_config.TextColumn("Formato"),
            "moneda": st.column_config.TextColumn("Moneda"),
            "destino": st.column_config.TextColumn("Destino"),
            "destino_nuevo": st.column_config.CheckboxColumn("Destino nuevo"),
            "cruce_moneda": st.column_config.CheckboxColumn("Cambio de moneda"),
            "cerca_umbral": st.column_config.CheckboxColumn("Bajo umbral"),
            "contribucion": st.column_config.ProgressColumn(
                "Contribucion", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "es_lavado": st.column_config.CheckboxColumn("Lavado (etiqueta)"),
        },
    )
    st.caption(
        "La columna `contribucion` combina el peso de atención de la etapa B con el error "
        "de reconstrucción de la etapa A, ambos normalizados dentro de la secuencia. "
        "`es_lavado` es la etiqueta real de cada transacción y no entra al modelo."
    )


if __name__ == "__main__":
    main()
