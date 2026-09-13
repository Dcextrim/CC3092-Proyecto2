"""Conversion de la tabla de variables a tensores de secuencias por remitente.

Cada remitente se convierte en una matriz de `longitud_maxima` pasos por
variable, alineada al inicio y completada con ceros. La máscara indica qué
pasos son transacciones reales y cuáles son relleno.

Uso:
    from data_pipeline.preprocessing import secuencias

    tensores = secuencias.tensorizar(tabla)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.preprocessing.features import VARIABLES
from utils.config import obtener


@dataclass
class Tensores:
    """Secuencias por remitente listas para alimentar a la red.

    Atributos:
        X: matriz de forma (remitentes, pasos, variables).
        mascara: booleana de forma (remitentes, pasos), True en transacciones reales.
        y: etiqueta por remitente, 1 si alguna transaccion de su ventana es lavado.
        longitud: numero de transacciones reales de cada remitente.
        identificador: cadena legible del remitente, por ejemplo `010_8000EBD30`.
        contexto: tabla larga con los atributos legibles de cada transaccion.
    """

    X: np.ndarray
    mascara: np.ndarray
    y: np.ndarray
    longitud: np.ndarray
    identificador: np.ndarray
    contexto: pd.DataFrame

    @property
    def variables(self) -> list[str]:
        """Nombres de las variables, en el orden del ultimo eje de `X`."""
        return list(VARIABLES)


CONTEXTO = [
    "fecha", "monto_usd", "formato_pago", "moneda_pagada",
    "banco_receptor", "cuenta_receptor", "es_lavado",
]


def tensorizar(
    tabla: pd.DataFrame,
    longitud_maxima: int | None = None,
    liberar: bool = False,
) -> Tensores:
    """Agrupa la tabla de variables en un tensor de secuencias por remitente.

    Espera la salida de `construir_variables`, ordenada por remitente y fecha.
    Con `liberar` activo las columnas de contexto se extraen de `tabla` en vez
    de copiarse, lo que evita duplicar cientos de megabytes y deja inutilizable
    la tabla recibida.
    """
    largo = longitud_maxima or obtener("secuencias.longitud_maxima", 32)

    emisor = tabla["emisor"].to_numpy()
    conteo = np.bincount(emisor)
    inicio = np.cumsum(conteo) - conteo
    posicion = (np.arange(len(emisor), dtype=np.int64) - np.repeat(inicio, conteo)).astype(np.int32)
    if posicion.max() >= largo:
        raise ValueError("Hay remitentes con mas transacciones que la longitud maxima")

    remitentes = len(conteo)
    banco = tabla["banco_emisor"].to_numpy()[inicio]
    cuenta = tabla["cuenta_emisor"].to_numpy()[inicio]

    tomar = (lambda nombre: tabla.pop(nombre)) if liberar else (lambda nombre: tabla[nombre])
    contexto = pd.DataFrame({"emisor": emisor, "paso": posicion})
    for nombre in CONTEXTO:
        contexto[nombre] = tomar(nombre).to_numpy()

    # Cada columna de `tabla` es una vista del bloque de variables, así que la
    # copia solo ocurre dentro del tensor. Descartar columnas aquí sería peor,
    # porque pandas reconstruiría el bloque completo en cada descarte.
    X = np.zeros((remitentes, largo, len(VARIABLES)), dtype=np.float32)
    for indice, nombre in enumerate(VARIABLES):
        X[emisor, posicion, indice] = tabla[nombre].to_numpy()

    mascara = np.zeros((remitentes, largo), dtype=bool)
    mascara[emisor, posicion] = True

    y = np.zeros(remitentes, dtype=np.int8)
    np.maximum.at(y, emisor, contexto["es_lavado"].to_numpy())

    identificador = np.array(
        [f"{int(b):05d}_{int(c):09X}" for b, c in zip(banco, cuenta)], dtype=object
    )

    return Tensores(
        X=X,
        mascara=mascara,
        y=y,
        longitud=conteo.astype(np.int32),
        identificador=identificador,
        contexto=contexto,
    )
