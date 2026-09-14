"""Clasificador supervisado construido sobre el codificador de la etapa A.

Reutiliza el mismo codificador que aprendió a representar el comportamiento
normal y le agrega una cabeza densa. Según la estrategia de transferencia
elegida, el codificador se congela o se ajusta con un learning rate menor que
el de la cabeza nueva.

Uso:
    modelo = Clasificador(n_variables=20, pesos_codificador=estado_etapa_a)
    modelo.aplicar_estrategia("ajuste_parcial")
    logit, atencion = modelo(X, mascara)
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from models.architectures.autoencoder import Codificador

ESTRATEGIAS = ("desde_cero", "extraccion", "ajuste_parcial", "ajuste_uniforme")


class Clasificador(nn.Module):
    """Codificador más cabeza densa que estima la probabilidad de lavado."""

    def __init__(
        self,
        n_variables: int,
        oculto: int = 64,
        latente: int = 128,
        oculto_cabeza: int = 64,
        dropout: float = 0.3,
        pesos_codificador: dict[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.codificador = Codificador(n_variables, oculto, latente)
        if pesos_codificador is not None:
            self.codificador.load_state_dict(pesos_codificador)
        self.cabeza = nn.Sequential(
            nn.Linear(latente, oculto_cabeza),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(oculto_cabeza, 1),
        )

    def forward(self, x: Tensor, mascara: Tensor) -> tuple[Tensor, Tensor]:
        """Devuelve el logit por secuencia y los pesos de atención."""
        z, atencion = self.codificador(x, mascara)
        return self.cabeza(z).squeeze(-1), atencion

    def aplicar_estrategia(self, estrategia: str) -> None:
        """Congela o libera el codificador según la estrategia de transferencia.

        `extraccion` congela el codificador y entrena solo la cabeza. Las demás
        lo dejan entrenable, y se distinguen por los pesos iniciales y por el
        learning rate. `desde_cero` parte de pesos aleatorios, `ajuste_parcial`
        parte de la etapa A con learning rate reducido, y `ajuste_uniforme`
        parte de la etapa A con el mismo learning rate que la cabeza.
        """
        if estrategia not in ESTRATEGIAS:
            raise ValueError(f"Estrategia desconocida: {estrategia}")
        entrenable = estrategia != "extraccion"
        for parametro in self.codificador.parameters():
            parametro.requires_grad = entrenable

    def grupos_parametros(
        self, learning_rate_codificador: float, learning_rate_cabeza: float
    ) -> list[dict[str, Iterable | float]]:
        """Arma los grupos para un optimizador con learning rate diferencial.

        La cabeza parte de una inicialización aleatoria y necesita pasos
        grandes. El codificador ya trae conocimiento útil y pasos grandes lo
        harían olvidarlo.
        """
        grupos = [{"params": list(self.cabeza.parameters()), "lr": learning_rate_cabeza}]
        entrenables = [p for p in self.codificador.parameters() if p.requires_grad]
        if entrenables:
            grupos.append({"params": entrenables, "lr": learning_rate_codificador})
        return grupos

    @torch.no_grad()
    def contar_parametros(self) -> dict[str, int]:
        """Parámetros totales y entrenables, por bloque."""
        contar = lambda modulo, solo_entrenables: sum(
            p.numel() for p in modulo.parameters() if p.requires_grad or not solo_entrenables
        )
        return {
            "codificador": contar(self.codificador, False),
            "codificador entrenable": contar(self.codificador, True),
            "cabeza": contar(self.cabeza, False),
        }
