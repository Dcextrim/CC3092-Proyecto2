"""Funciones de pérdida para clasificación con desbalance extremo.

Con 72 remitentes normales por cada sospechoso, la entropía cruzada sin ajuste
queda dominada por los ejemplos fáciles de la clase mayoritaria. Las dos
opciones implementadas atacan ese problema por caminos distintos, y el segundo
cuaderno las compara de forma empírica.

Uso:
    perdida = PerdidaFocal(gamma=2.0, alpha=0.75)
    valor = perdida(logits, etiquetas)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PerdidaFocal(nn.Module):
    """Entropía cruzada binaria ponderada por la dificultad del ejemplo.

    El factor `(1 - p)**gamma` reduce el peso de los ejemplos que el modelo ya
    acierta con holgura, de modo que el gradiente se concentra en los difíciles.
    El factor `alpha` compensa además la frecuencia de cada clase.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, etiquetas: Tensor) -> Tensor:
        """Pérdida promedio del lote."""
        etiquetas = etiquetas.float()
        entropia = F.binary_cross_entropy_with_logits(logits, etiquetas, reduction="none")
        probabilidad = torch.sigmoid(logits)
        acierto = probabilidad * etiquetas + (1 - probabilidad) * (1 - etiquetas)
        peso_clase = self.alpha * etiquetas + (1 - self.alpha) * (1 - etiquetas)
        return (peso_clase * (1 - acierto) ** self.gamma * entropia).mean()


class EntropiaPonderada(nn.Module):
    """Entropía cruzada binaria con peso fijo sobre la clase positiva.

    Es la alternativa clásica a la pérdida focal. Sube el costo de equivocarse
    en un positivo en proporción a lo raro que es, sin distinguir entre
    ejemplos fáciles y difíciles.
    """

    def __init__(self, peso_positivo: float) -> None:
        super().__init__()
        self.register_buffer("peso_positivo", torch.tensor(float(peso_positivo)))

    def forward(self, logits: Tensor, etiquetas: Tensor) -> Tensor:
        """Pérdida promedio del lote."""
        return F.binary_cross_entropy_with_logits(
            logits, etiquetas.float(), pos_weight=self.peso_positivo
        )
