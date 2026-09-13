"""Partición por remitente y normalización sin filtración entre conjuntos.

La unidad de partición es el remitente completo, nunca la transacción, de modo
que una misma secuencia no puede aparecer en dos conjuntos. Las estadísticas de
normalización se estiman solo con el conjunto de entrenamiento.

Uso:
    from data_pipeline.preprocessing import splits

    particion = splits.particionar(tensores.y)
    escala = splits.ajustar_escala(tensores, particion.entrenamiento)
    splits.aplicar_escala(tensores, escala)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from data_pipeline.preprocessing.features import VARIABLES
from data_pipeline.preprocessing.secuencias import Tensores
from utils.config import obtener

CONTINUAS = ["log_monto", "log_delta_t", "log_repeticiones_destino", "desvio_monto"]


@dataclass
class Particion:
    """Índices de remitentes en cada conjunto."""

    entrenamiento: np.ndarray
    validacion: np.ndarray
    prueba: np.ndarray

    def resumen(self, y: np.ndarray) -> dict[str, dict[str, float]]:
        """Tamaño, positivos y prevalencia de cada conjunto."""
        return {
            nombre: {
                "remitentes": int(len(indices)),
                "positivos": int(y[indices].sum()),
                "prevalencia": float(y[indices].mean()),
            }
            for nombre, indices in (
                ("entrenamiento", self.entrenamiento),
                ("validacion", self.validacion),
                ("prueba", self.prueba),
            )
        }


@dataclass
class Escala:
    """Media y desviación por variable continua, estimadas en entrenamiento."""

    columnas: list[int]
    media: np.ndarray
    desviacion: np.ndarray


def particionar(y: np.ndarray, semilla: int | None = None) -> Particion:
    """Divide los remitentes en entrenamiento, validación y prueba de forma estratificada."""
    semilla = semilla if semilla is not None else obtener("particion.semilla", 42)
    fraccion_val = float(obtener("particion.validacion", 0.15))
    fraccion_prueba = float(obtener("particion.prueba", 0.15))

    indices = np.arange(len(y))
    resto, prueba = train_test_split(
        indices, test_size=fraccion_prueba, stratify=y, random_state=semilla
    )
    proporcion_val = fraccion_val / (1.0 - fraccion_prueba)
    entrenamiento, validacion = train_test_split(
        resto, test_size=proporcion_val, stratify=y[resto], random_state=semilla
    )
    return Particion(
        entrenamiento=np.sort(entrenamiento),
        validacion=np.sort(validacion),
        prueba=np.sort(prueba),
    )


def ajustar_escala(tensores: Tensores, entrenamiento: np.ndarray) -> Escala:
    """Estima media y desviación de las variables continuas sobre pasos reales.

    Las variables binarias y las codificaciones circulares ya están acotadas, así
    que se dejan intactas.
    """
    columnas = [VARIABLES.index(nombre) for nombre in CONTINUAS]
    reales = np.zeros(tensores.mascara.shape, dtype=bool)
    reales[entrenamiento] = tensores.mascara[entrenamiento]

    media = np.empty(len(columnas), dtype=np.float64)
    desviacion = np.empty(len(columnas), dtype=np.float64)
    for indice, columna in enumerate(columnas):
        plano = tensores.X[:, :, columna][reales]
        media[indice] = plano.mean()
        desviacion[indice] = plano.std()
    desviacion[desviacion < 1e-6] = 1.0
    return Escala(columnas=columnas, media=media, desviacion=desviacion)


def aplicar_escala(tensores: Tensores, escala: Escala) -> None:
    """Normaliza las variables continuas de `tensores.X` en el lugar."""
    for indice, columna in enumerate(escala.columnas):
        vista = tensores.X[:, :, columna]
        vista -= escala.media[indice]
        vista /= escala.desviacion[indice]
        vista[~tensores.mascara] = 0.0
