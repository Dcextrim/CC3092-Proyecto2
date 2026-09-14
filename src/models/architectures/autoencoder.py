"""Autoencoder secuencial con atención para aprender el comportamiento normal.

El codificador comprime una secuencia de longitud variable en un vector, y el
decodificador intenta reconstruir la secuencia original a partir de ese vector.
Lo que el modelo no logre reconstruir es la señal de anomalía.

La atención vive dentro del codificador y no entre codificador y decodificador.
Una atención cruzada dejaria que el decodificador copiara la entrada paso a
paso, y el error de reconstrucción perderia su valor como señal.

Uso:
    modelo = AutoencoderSecuencial(n_variables=20, longitud=32)
    reconstruccion, z, atencion = modelo(X, mascara)
    error = error_por_secuencia(X, reconstruccion, mascara)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENOS_INFINITO = -1e9


class AtencionAditiva(nn.Module):
    """Atención aditiva enmáscarada sobre los pasos de la secuencia.

    Calcula un peso por paso con `v` aplicado a la tangente hiperbólica de una
    proyección lineal, y devuelve el promedio ponderado junto con los pesos.
    """

    def __init__(self, dimension: int, interna: int = 64) -> None:
        super().__init__()
        self.proyeccion = nn.Linear(dimension, interna)
        self.puntaje = nn.Linear(interna, 1, bias=False)

    def forward(self, estados: Tensor, mascara: Tensor) -> tuple[Tensor, Tensor]:
        """Devuelve el contexto de forma (lote, dimension) y los pesos (lote, pasos)."""
        puntajes = self.puntaje(torch.tanh(self.proyeccion(estados))).squeeze(-1)
        puntajes = puntajes.masked_fill(~mascara, MENOS_INFINITO)
        pesos = torch.softmax(puntajes, dim=1)
        contexto = torch.bmm(pesos.unsqueeze(1), estados).squeeze(1)
        return contexto, pesos


class Codificador(nn.Module):
    """Proyección lineal, GRU bidireccional, normalización y atención."""

    def __init__(self, n_variables: int, oculto: int = 64, latente: int = 128) -> None:
        super().__init__()
        self.entrada = nn.Linear(n_variables, oculto)
        self.gru = nn.GRU(oculto, oculto, batch_first=True, bidirectional=True)
        self.norma = nn.LayerNorm(2 * oculto)
        self.atencion = AtencionAditiva(2 * oculto)
        self.proyeccion = nn.Linear(2 * oculto, latente)

    def forward(self, x: Tensor, mascara: Tensor) -> tuple[Tensor, Tensor]:
        """Devuelve el vector comprimido y los pesos de atención."""
        estados, _ = self.gru(torch.relu(self.entrada(x)))
        estados = self.norma(estados)
        contexto, pesos = self.atencion(estados, mascara)
        return self.proyeccion(contexto), pesos


class Decodificador(nn.Module):
    """Repite el vector comprimido en el tiempo y reconstruye las variables."""

    def __init__(self, n_variables: int, oculto: int = 64, latente: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(latente, oculto, batch_first=True)
        self.salida = nn.Linear(oculto, n_variables)

    def forward(self, z: Tensor, longitud: int) -> Tensor:
        """Devuelve la reconstrucción de forma (lote, pasos, variables)."""
        estados, _ = self.gru(z.unsqueeze(1).expand(-1, longitud, -1))
        return self.salida(estados)


class AutoencoderSecuencial(nn.Module):
    """Codificador y decodificador acoplados, la etapa A del sistema."""

    def __init__(
        self,
        n_variables: int,
        longitud: int,
        oculto: int = 64,
        latente: int = 128,
    ) -> None:
        super().__init__()
        self.longitud = longitud
        self.codificador = Codificador(n_variables, oculto, latente)
        self.decodificador = Decodificador(n_variables, oculto, latente)

    def forward(self, x: Tensor, mascara: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Devuelve reconstrucción, vector comprimido y pesos de atención."""
        z, atencion = self.codificador(x, mascara)
        return self.decodificador(z, x.shape[1]), z, atencion


def error_por_paso(x: Tensor, reconstruccion: Tensor, mascara: Tensor) -> Tensor:
    """Error cuadrático medio de cada paso, con cero en el relleno."""
    return ((x - reconstruccion) ** 2).mean(dim=-1) * mascara


def error_por_secuencia(x: Tensor, reconstruccion: Tensor, mascara: Tensor) -> Tensor:
    """Error cuadrático medio de cada secuencia sobre sus pasos reales.

    Se promedia sobre los pasos reales y no sobre los 32 posibles, para que una
    secuencia corta no obtenga un error artificialmente bajo por su relleno.
    """
    pasos = mascara.sum(dim=1).clamp(min=1)
    return error_por_paso(x, reconstruccion, mascara).sum(dim=1) / pasos
