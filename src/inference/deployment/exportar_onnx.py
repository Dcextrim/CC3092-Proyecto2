"""Exportación del sistema completo a ONNX para el MVP.

El modelo exportado devuelve en una sola llamada todo lo que la interfaz
necesita, la probabilidad combinada, la probabilidad de la etapa B, el error de
reconstrucción total y por paso, y los pesos de atención.

Uso:
    python -m inference.deployment.exportar_onnx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from models.architectures.autoencoder import (
    AutoencoderSecuencial,
    error_por_paso,
    error_por_secuencia,
)
from models.architectures.clasificador import Clasificador
from utils.rutas import RESULTS_MODELS

NOMBRE = "sistema.onnx"
OPSET = 17
SALIDAS = [
    "probabilidad", "probabilidad_etapa_b", "error_reconstruccion",
    "error_por_paso", "atencion",
]


class SistemaCompleto(nn.Module):
    """Une ambas etapas y la fusión en un solo grafo exportable.

    Los coeficientes de la regresión logística de fusión se incorporan como
    constantes, de modo que el archivo ONNX contiene el sistema entero y la
    aplicación no necesita reimplementar ninguna fórmula.
    """

    def __init__(
        self,
        autoencoder: AutoencoderSecuencial,
        clasificador: Clasificador,
        media_error: float,
        desviacion_error: float,
        coeficientes: np.ndarray,
        intercepto: float,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.clasificador = clasificador
        self.register_buffer("media_error", torch.tensor(float(media_error)))
        self.register_buffer("desviacion_error", torch.tensor(float(desviacion_error)))
        self.register_buffer("coeficientes", torch.tensor(np.asarray(coeficientes, dtype=np.float32).ravel()))
        self.register_buffer("intercepto", torch.tensor(float(intercepto)))

    def forward(self, x: Tensor, mascara: Tensor) -> tuple[Tensor, ...]:
        """Devuelve las cinco salidas que consume la interfaz."""
        booleana = mascara > 0.5
        reconstruccion, _, _ = self.autoencoder(x, booleana)
        por_paso = error_por_paso(x, reconstruccion, booleana)
        error = error_por_secuencia(x, reconstruccion, booleana)

        logit, atencion = self.clasificador(x, booleana)
        normalizado = (error - self.media_error) / self.desviacion_error
        combinado = (
            normalizado * self.coeficientes[0] + logit * self.coeficientes[1] + self.intercepto
        )
        return torch.sigmoid(combinado), torch.sigmoid(logit), error, por_paso, atencion


def exportar(
    sistema: SistemaCompleto,
    longitud: int,
    n_variables: int,
    destino: Path | None = None,
) -> Path:
    """Escribe el grafo ONNX con lote y longitud dinámicos."""
    destino = destino or (RESULTS_MODELS / NOMBRE)
    destino.parent.mkdir(parents=True, exist_ok=True)
    sistema = sistema.cpu().eval()

    ejemplo_x = torch.randn(2, longitud, n_variables)
    ejemplo_mascara = torch.ones(2, longitud)

    torch.onnx.export(
        sistema,
        (ejemplo_x, ejemplo_mascara),
        str(destino),
        input_names=["x", "mascara"],
        output_names=SALIDAS,
        dynamic_axes={
            "x": {0: "lote", 1: "pasos"},
            "mascara": {0: "lote", 1: "pasos"},
            **{nombre: {0: "lote"} for nombre in SALIDAS},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    return destino


def verificar(
    sistema: SistemaCompleto,
    ruta: Path,
    x: np.ndarray,
    mascara: np.ndarray,
    tolerancia: float = 1e-4,
) -> dict[str, float]:
    """Compara las salidas de ONNX contra las de PyTorch y devuelve la diferencia máxima."""
    import onnxruntime

    sistema = sistema.cpu().eval()
    with torch.no_grad():
        esperado = sistema(torch.from_numpy(x), torch.from_numpy(mascara.astype(np.float32)))

    sesion = onnxruntime.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    obtenido = sesion.run(None, {"x": x, "mascara": mascara.astype(np.float32)})

    diferencias = {
        nombre: float(np.abs(a.numpy() - b).max())
        for nombre, a, b in zip(SALIDAS, esperado, obtenido)
    }
    peor = max(diferencias.values())
    if peor > tolerancia:
        raise AssertionError(f"ONNX difiere de PyTorch en {peor:.2e}: {diferencias}")
    return diferencias
