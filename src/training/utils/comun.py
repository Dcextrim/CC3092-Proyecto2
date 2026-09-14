"""Utilidades compartidas por los entrenamientos de ambas etapas.

Uso:
    from training.utils.comun import fijar_semilla, dispositivo, lotes
"""

from __future__ import annotations

import random
from typing import Iterator

import numpy as np
import torch


def dispositivo() -> torch.device:
    """Devuelve la GPU si esta disponible, y la CPU en caso contrario."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(semilla: int) -> torch.Generator:
    """Fija las semillas de Python, NumPy y PyTorch, y devuelve un generador."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    generador = torch.Generator()
    generador.manual_seed(semilla)
    return generador


def lotes(
    indices: torch.Tensor,
    tamano: int,
    mezclar: bool = False,
    generador: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    """Recorre `indices` en lotes, opcionalmente en orden aleatorio."""
    if mezclar:
        indices = indices[torch.randperm(len(indices), generator=generador)]
    for inicio in range(0, len(indices), tamano):
        yield indices[inicio : inicio + tamano]


def submuestrear_negativos(
    y: np.ndarray, indices: np.ndarray, razon: int, semilla: int
) -> np.ndarray:
    """Conserva todos los positivos y `razon` negativos por cada uno.

    Se aplica solo al conjunto de entrenamiento. La evaluación siempre ocurre
    sobre la distribución natural, porque de otro modo la precisión reportada no
    tendría relación con la que vería un equipo de cumplimiento.
    """
    positivos = indices[y[indices] == 1]
    negativos = indices[y[indices] == 0]
    cuantos = min(len(negativos), razon * len(positivos))
    elegidos = np.random.default_rng(semilla).choice(negativos, size=cuantos, replace=False)
    return np.sort(np.concatenate([positivos, elegidos]))
