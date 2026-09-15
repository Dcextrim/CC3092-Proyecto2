"""Vista legible de la secuencia de un remitente.

Traduce los códigos internos del pipeline a los nombres que un analista
reconoce, y recupera desde el tensor los indicadores binarios que la explicación
y el mapa de calor necesitan.

Uso:
    from inference.utils.vista import contexto_remitente

    tabla = contexto_remitente(contexto, 0, X[0], mascara[0], variables, catalogos)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INDICADORES = ("destino_nuevo", "cruce_moneda", "cerca_umbral", "cruce_banco", "auto_transferencia")


def contexto_remitente(
    contexto: pd.DataFrame,
    remitente: int,
    X: np.ndarray,
    mascara: np.ndarray,
    variables: list[str],
    catalogos: dict[str, list[str]],
    columna_remitente: str = "remitente",
) -> pd.DataFrame:
    """Devuelve una fila por transacción del remitente, con nombres legibles.

    `X` y `mascara` son la secuencia de ese remitente, de forma (pasos,
    variables) y (pasos,).
    """
    filas = contexto[contexto[columna_remitente] == remitente].sort_values("paso")
    pasos = int(mascara.sum())

    tabla = pd.DataFrame({
        "paso": filas["paso"].to_numpy()[:pasos],
        "fecha": pd.to_datetime(filas["fecha"].to_numpy()[:pasos], unit="s"),
        "monto_usd": filas["monto_usd"].to_numpy()[:pasos],
        "formato": [catalogos["formatos"][c] for c in filas["formato_pago"].to_numpy()[:pasos]],
        "moneda": [catalogos["monedas"][c] for c in filas["moneda_pagada"].to_numpy()[:pasos]],
        "destino": [
            f"{int(banco):05d}_{int(cuenta):09X}"
            for banco, cuenta in zip(
                filas["banco_receptor"].to_numpy()[:pasos],
                filas["cuenta_receptor"].to_numpy()[:pasos],
            )
        ],
        "es_lavado": filas["es_lavado"].to_numpy()[:pasos],
    })
    for indicador in INDICADORES:
        tabla[indicador] = X[:pasos, variables.index(indicador)].astype(int)
    return tabla
