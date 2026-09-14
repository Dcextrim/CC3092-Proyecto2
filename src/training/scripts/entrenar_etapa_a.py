"""Entrenamiento de la etapa A, el aprendizaje de la normalidad.

El autoencoder se entrena exclusivamente con remitentes normales del conjunto
de entrenamiento. Nunca ve una secuencia etiquetada como lavado, y el código lo
comprueba antes de empezar. Lo que después no logre reconstruir es, por
construcción, comportamiento que no se parece a lo "normal".

Uso:
    from training.scripts.entrenar_etapa_a import entrenar_autoencoder, puntuar

    modelo, historial = entrenar_autoencoder(datos, semilla=0)
    puntajes = puntuar(modelo, datos.tensores)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from data_pipeline.preprocessing.features import VARIABLES
from data_pipeline.preprocessing.secuencias import Tensores
from models.architectures.autoencoder import (
    AutoencoderSecuencial,
    error_por_paso,
    error_por_secuencia,
)
from training.utils.comun import dispositivo as dispositivo_disponible
from training.utils.comun import fijar_semilla, lotes
from utils.config import obtener


@dataclass
class Historial:
    """Pérdida por época y época en la que se detuvo el entrenamiento."""

    entrenamiento: list[float] = field(default_factory=list)
    validacion: list[float] = field(default_factory=list)
    mejor_epoca: int = 0
    segundos: float = 0.0

    @property
    def mejor_perdida(self) -> float:
        """Pérdida de validación en la mejor época."""
        return self.validacion[self.mejor_epoca]


@dataclass
class Puntajes:
    """Salidas de la etapa A para un conjunto de secuencias."""

    error: np.ndarray
    error_por_paso: np.ndarray
    atencion: np.ndarray
    latente: np.ndarray | None = None


def _tensores_torch(tensores: Tensores) -> tuple[torch.Tensor, torch.Tensor]:
    """Envuelve los arreglos de NumPy sin copiarlos."""
    return torch.from_numpy(tensores.X), torch.from_numpy(tensores.mascara)


def _perdida_epoca(
    modelo: nn.Module,
    X: torch.Tensor,
    mascara: torch.Tensor,
    indices: torch.Tensor,
    equipo: torch.device,
    tamano_lote: int,
) -> float:
    """Pérdida media de reconstrucción sobre un conjunto, sin actualizar pesos."""
    modelo.eval()
    total, vistos = 0.0, 0
    with torch.no_grad():
        for lote in lotes(indices, tamano_lote):
            x = X[lote].to(equipo)
            m = mascara[lote].to(equipo)
            reconstruccion, _, _ = modelo(x, m)
            total += float(error_por_secuencia(x, reconstruccion, m).sum())
            vistos += len(lote)
    return total / vistos


def entrenar_autoencoder(
    datos,
    semilla: int = 0,
    equipo: torch.device | None = None,
    limite_entrenamiento: int | None = None,
    verboso: bool = True,
) -> tuple[AutoencoderSecuencial, Historial]:
    """Entrena el autoencoder sobre los remitentes normales de entrenamiento.

    `limite_entrenamiento` recorta el conjunto para pruebas rápidas y no debe
    usarse para los resultados que se reportan.
    """
    equipo = equipo or dispositivo_disponible()
    generador = fijar_semilla(semilla)
    tensores = datos.tensores

    normales_entrenamiento = datos.particion.entrenamiento[
        tensores.y[datos.particion.entrenamiento] == 0
    ]
    normales_validacion = datos.particion.validacion[
        tensores.y[datos.particion.validacion] == 0
    ]
    if tensores.y[normales_entrenamiento].sum() or tensores.y[normales_validacion].sum():
        raise AssertionError("La etapa A recibio secuencias etiquetadas como lavado")

    if limite_entrenamiento:
        normales_entrenamiento = normales_entrenamiento[:limite_entrenamiento]
        normales_validacion = normales_validacion[: max(1, limite_entrenamiento // 5)]

    X, mascara = _tensores_torch(tensores)
    indices_entrenamiento = torch.from_numpy(normales_entrenamiento.astype(np.int64))
    indices_validacion = torch.from_numpy(normales_validacion.astype(np.int64))

    modelo = AutoencoderSecuencial(
        n_variables=len(VARIABLES),
        longitud=X.shape[1],
        oculto=obtener("etapa_a.dimension_oculta", 64),
        latente=obtener("etapa_a.dimension_latente", 128),
    ).to(equipo)

    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=obtener("etapa_a.learning_rate", 1e-3),
        weight_decay=obtener("etapa_a.weight_decay", 1e-4),
    )
    recorte = obtener("etapa_a.recorte_gradiente", 1.0)
    tamano_lote = obtener("entrenamiento.batch", 512)
    paciencia = obtener("etapa_a.paciencia", 4)
    epocas = obtener("etapa_a.epocas_maximas", 30)

    historial = Historial()
    mejor_estado = copy.deepcopy(modelo.state_dict())
    mejor_perdida = float("inf")
    sin_mejora = 0
    inicio = time.time()

    for epoca in range(epocas):
        modelo.train()
        acumulado, vistos = 0.0, 0
        for lote in lotes(indices_entrenamiento, tamano_lote, mezclar=True, generador=generador):
            x = X[lote].to(equipo)
            m = mascara[lote].to(equipo)
            reconstruccion, _, _ = modelo(x, m)
            perdida = error_por_secuencia(x, reconstruccion, m).mean()

            optimizador.zero_grad(set_to_none=True)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), recorte)
            optimizador.step()

            acumulado += float(perdida.detach()) * len(lote)
            vistos += len(lote)

        perdida_entrenamiento = acumulado / vistos
        perdida_validacion = _perdida_epoca(
            modelo, X, mascara, indices_validacion, equipo, tamano_lote
        )
        historial.entrenamiento.append(perdida_entrenamiento)
        historial.validacion.append(perdida_validacion)

        if perdida_validacion < mejor_perdida - 1e-5:
            mejor_perdida, sin_mejora = perdida_validacion, 0
            historial.mejor_epoca = epoca
            mejor_estado = copy.deepcopy(modelo.state_dict())
        else:
            sin_mejora += 1

        if verboso:
            marca = "  <- mejor" if sin_mejora == 0 else ""
            print(f"  epoca {epoca + 1:>2}  entrenamiento {perdida_entrenamiento:.5f}"
                  f"  validacion {perdida_validacion:.5f}{marca}", flush=True)

        if sin_mejora >= paciencia:
            if verboso:
                print(f"  parada temprana tras {epoca + 1} epocas", flush=True)
            break

    modelo.load_state_dict(mejor_estado)
    historial.segundos = time.time() - inicio
    return modelo, historial


@torch.no_grad()
def puntuar(
    modelo: AutoencoderSecuencial,
    tensores: Tensores,
    equipo: torch.device | None = None,
    tamano_lote: int = 1024,
    incluir_latente: bool = False,
) -> Puntajes:
    """Calcula error de reconstruccion, error por paso y atención."""
    equipo = equipo or dispositivo_disponible()
    modelo = modelo.to(equipo).eval()
    X, mascara = _tensores_torch(tensores)
    total = len(tensores.y)

    error = np.zeros(total, dtype=np.float32)
    por_paso = np.zeros(tensores.mascara.shape, dtype=np.float32)
    atencion = np.zeros(tensores.mascara.shape, dtype=np.float32)
    latente = (
        np.zeros((total, modelo.codificador.proyeccion.out_features), dtype=np.float32)
        if incluir_latente else None
    )

    for lote in lotes(torch.arange(total), tamano_lote):
        x = X[lote].to(equipo)
        m = mascara[lote].to(equipo)
        reconstruccion, z, pesos = modelo(x, m)
        error[lote] = error_por_secuencia(x, reconstruccion, m).cpu().numpy()
        por_paso[lote] = error_por_paso(x, reconstruccion, m).cpu().numpy()
        atencion[lote] = pesos.cpu().numpy()
        if latente is not None:
            latente[lote] = z.cpu().numpy()

    return Puntajes(error=error, error_por_paso=por_paso, atencion=atencion, latente=latente)
