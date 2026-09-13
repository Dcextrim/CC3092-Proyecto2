"""Estilo común de los gráficos y vistas de secuencias.

Uso:
    from utils.visualizacion import set_estilo, mapa_secuencia

    set_estilo()
    mapa_secuencia(tensores, indice=0)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.rutas import RESULTS_PLOTS, asegurar_carpetas

PALETA = "deep"
COLOR_NORMAL = "#4C72B0"
COLOR_SOSPECHOSO = "#C44E52"


def set_estilo() -> None:
    """Fija el tema de seaborn y los parámetros comunes de matplotlib."""
    sns.set_theme(style="whitegrid", palette=PALETA)
    plt.rcParams.update({
        "figure.figsize": (9, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
    })


def guardar(nombre: str, figura: plt.Figure | None = None) -> None:
    """Escribe la figura activa en `results/plots/<nombre>.png`."""
    asegurar_carpetas()
    (figura or plt.gcf()).savefig(RESULTS_PLOTS / f"{nombre}.png")


def mapa_secuencia(
    X: np.ndarray,
    mascara: np.ndarray,
    variables: list[str],
    titulo: str,
    eje: plt.Axes | None = None,
    limite: float = 3.0,
    barra: bool = True,
) -> plt.Axes:
    """Dibuja una secuencia como mapa de calor de variables por paso temporal.

    Los pasos de relleno se dejan en blanco para distinguirlos de los ceros
    reales. La escala de color se fija en `limite` para que varios mapas puestos
    lado a lado sean comparables entre sí.
    """
    eje = eje or plt.gca()
    datos = np.where(mascara[None, :], X.T, np.nan)
    sns.heatmap(
        datos,
        ax=eje,
        cmap="RdBu_r",
        vmin=-limite,
        vmax=limite,
        center=0,
        yticklabels=variables,
        xticklabels=5,
        cbar=barra,
        cbar_kws={"label": "valor normalizado"} if barra else None,
    )
    eje.set_title(titulo)
    eje.set_xlabel("paso de la secuencia")
    eje.tick_params(axis="y", labelsize=7)
    return eje


def mapa_contribucion(
    atencion: np.ndarray,
    anomalia: np.ndarray,
    titulo: str,
    eje: plt.Axes | None = None,
) -> plt.Axes:
    """Mapa de calor de la contribución de cada transacción a una alerta.

    La primera fila es el peso de atención que la etapa B dió a cada paso, y la
    segunda es el error de reconstrucción de la etapa A. Ambas se llevan al
    rango cero a uno dentro de la secuencia.
    """
    eje = eje or plt.gca()
    normalizar = lambda v: (
        (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.zeros_like(v)
    )
    datos = np.vstack([normalizar(atencion), normalizar(anomalia)])
    sns.heatmap(
        datos,
        ax=eje,
        cmap="rocket_r",
        vmin=0,
        vmax=1,
        yticklabels=["atencion (etapa B)", "anomalia (etapa A)"],
        xticklabels=[str(i + 1) for i in range(datos.shape[1])],
        cbar_kws={"label": "contribucion relativa"},
        linewidths=0.4,
        linecolor="white",
    )
    eje.set_title(titulo, fontsize=11)
    eje.set_xlabel("transaccion de la secuencia")
    eje.tick_params(axis="y", rotation=0, labelsize=8)
    return eje
