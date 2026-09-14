"""Entrenamiento de la etapa B, la clasificación supervisada con transferencia.

El clasificador reutiliza el codificador que la etapa A entrenó sin etiquetas.
Se comparan tres estrategias de la semana 9, entrenar desde cero, congelar el
codificador y usarlo como extractor de rasgos, y ajustarlo parcialmente con un
learning rate menor que el de la cabeza nueva.

Uso:
    from training.scripts.entrenar_etapa_b import entrenar_clasificador, predecir

    modelo, historial = entrenar_clasificador(datos, "ajuste_parcial", pesos)
    probabilidad, atencion = predecir(modelo, datos.tensores)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import Tensor, nn

from data_pipeline.preprocessing.features import VARIABLES
from data_pipeline.preprocessing.secuencias import Tensores
from models.architectures.clasificador import Clasificador
from models.losses.focal import EntropiaPonderada, PerdidaFocal
from training.utils.comun import dispositivo as dispositivo_disponible
from training.utils.comun import fijar_semilla, lotes, submuestrear_negativos
from utils.config import obtener

PERDIDAS = ("focal", "ponderada")

def _reducir_positivos(
    y: np.ndarray, indices: np.ndarray, fraccion: float, semilla: int
) -> np.ndarray:
    """Conserva solo una fracción de los positivos de entrenamiento.

    Sirve para medir cuanta etiqueta hace falta antes de que la transferencia
    deje de aportar. Los negativos se dejan intactos, porque el submuestreo
    posterior los ajusta a la nueva cantidad de positivos.
    """
    if fraccion >= 1.0:
        return indices
    positivos = indices[y[indices] == 1]
    negativos = indices[y[indices] == 0]
    cuantos = max(1, int(round(fraccion * len(positivos))))
    elegidos = np.random.default_rng(1000 + semilla).choice(positivos, size=cuantos, replace=False)
    return np.sort(np.concatenate([elegidos, negativos]))


def _muestra_de_seguimiento(y: np.ndarray, indices: np.ndarray, semilla: int) -> np.ndarray:
    """Conjunto sobre el que se vigila el PR-AUC entre épocas.

    Por omisión es la validación completa. Con `negativos_de_seguimiento` mayor
    que cero se usa una muestra que conserva todos los positivos, lo que abarata
    el seguimiento en máquinas sin GPU a cambio de una elección de época algo
    más ruidosa. Las métricas que se reportan siempre se calculan sobre los
    conjuntos completos.
    """
    cupo = int(obtener("etapa_b.negativos_de_seguimiento", 0) or 0)
    if cupo <= 0:
        return indices
    positivos = indices[y[indices] == 1]
    negativos = indices[y[indices] == 0]
    cuantos = min(len(negativos), cupo)
    elegidos = np.random.default_rng(semilla).choice(negativos, size=cuantos, replace=False)
    return np.sort(np.concatenate([positivos, elegidos]))


@dataclass
class HistorialB:
    """Pérdida de entrenamiento y PR-AUC de validación por época."""

    perdida: list[float] = field(default_factory=list)
    pr_auc_validacion: list[float] = field(default_factory=list)
    mejor_epoca: int = 0
    segundos: float = 0.0

    @property
    def mejor_pr_auc(self) -> float:
        """PR-AUC de validación en la mejor época."""
        return self.pr_auc_validacion[self.mejor_epoca]


def construir_perdida(tipo: str, razon_negativos: int) -> nn.Module:
    """Devuelve la función de pérdida pedida, configurada para el desbalance."""
    if tipo == "focal":
        return PerdidaFocal(
            gamma=obtener("etapa_b.focal_gamma", 2.0),
            alpha=obtener("etapa_b.focal_alpha", 0.75),
        )
    if tipo == "ponderada":
        return EntropiaPonderada(peso_positivo=float(razon_negativos))
    raise ValueError(f"Perdida desconocida: {tipo}")


@torch.no_grad()
def _logits(
    modelo: Clasificador,
    X: Tensor,
    mascara: Tensor,
    indices: Tensor,
    equipo: torch.device,
    tamano_lote: int = 1024,
) -> np.ndarray:
    """Logits del modelo sobre un subconjunto de secuencias."""
    modelo.eval()
    salida = np.zeros(len(indices), dtype=np.float32)
    posicion = 0
    for lote in lotes(indices, tamano_lote):
        valores, _ = modelo(X[lote].to(equipo), mascara[lote].to(equipo))
        salida[posicion : posicion + len(lote)] = valores.cpu().numpy()
        posicion += len(lote)
    return salida


def entrenar_clasificador(
    datos,
    estrategia: str,
    pesos_codificador: dict[str, Tensor] | None = None,
    tipo_perdida: str = "focal",
    semilla: int = 0,
    equipo: torch.device | None = None,
    verboso: bool = True,
    limite_entrenamiento: int | None = None,
    fraccion_positivos: float = 1.0,
) -> tuple[Clasificador, HistorialB]:
    """Entrena la cabeza de clasificación según la estrategia de transferencia.

    `estrategia` es `desde_cero`, `extraccion` o `ajuste_parcial`. Las dos
    ultimas requieren `pesos_codificador` provenientes de la etapa A.

    La selección del mejor modelo usa PR-AUC sobre el conjunto de validación,
    que es la distribución que el sistema verá en operación. Las métricas que se
    reportan se calculan después sobre los conjuntos completos.
    """
    if estrategia != "desde_cero" and pesos_codificador is None:
        raise ValueError(f"La estrategia {estrategia} necesita pesos de la etapa A")

    equipo = equipo or dispositivo_disponible()
    generador = fijar_semilla(semilla)
    tensores = datos.tensores
    razon = obtener("etapa_b.razon_negativos", 20)

    disponibles = _reducir_positivos(
        tensores.y, datos.particion.entrenamiento, fraccion_positivos, semilla
    )
    entrenamiento = submuestrear_negativos(tensores.y, disponibles, razon, semilla)
    if limite_entrenamiento:
        entrenamiento = entrenamiento[:limite_entrenamiento]
    validacion = datos.particion.validacion
    seguimiento = _muestra_de_seguimiento(tensores.y, validacion, semilla)

    X = torch.from_numpy(tensores.X)
    mascara = torch.from_numpy(tensores.mascara)
    y = torch.from_numpy(tensores.y.astype(np.float32))
    indices_entrenamiento = torch.from_numpy(entrenamiento.astype(np.int64))
    indices_seguimiento = torch.from_numpy(seguimiento.astype(np.int64))
    y_seguimiento = tensores.y[seguimiento]

    modelo = Clasificador(
        n_variables=len(VARIABLES),
        oculto=obtener("etapa_a.dimension_oculta", 64),
        latente=obtener("etapa_a.dimension_latente", 128),
        pesos_codificador=pesos_codificador if estrategia != "desde_cero" else None,
    ).to(equipo)
    modelo.aplicar_estrategia(estrategia)

    # Solo `ajuste_parcial` usa un learning rate menor en el codificador. Las
    # demás estrategias lo entrenan al mismo ritmo que la cabeza, lo que permite
    # separar el efecto de la inicialización del efecto del learning rate.
    learning_rate_codificador = (
        obtener("etapa_b.learning_rate_codificador", 1e-4)
        if estrategia == "ajuste_parcial"
        else obtener("etapa_b.learning_rate_cabeza", 1e-3)
    )
    optimizador = torch.optim.AdamW(
        modelo.grupos_parametros(
            learning_rate_codificador, obtener("etapa_b.learning_rate_cabeza", 1e-3)
        ),
        weight_decay=obtener("etapa_b.weight_decay", 1e-4),
    )
    perdida_fn = construir_perdida(tipo_perdida, razon).to(equipo)
    recorte = obtener("etapa_b.recorte_gradiente", 1.0)
    tamano_lote = obtener("entrenamiento.batch", 512)
    paciencia = obtener("etapa_b.paciencia", 5)
    epocas = obtener("etapa_b.epocas_maximas", 25)

    historial = HistorialB()
    mejor_estado = copy.deepcopy(modelo.state_dict())
    mejor_pr_auc = -1.0
    sin_mejora = 0
    inicio = time.time()

    for epoca in range(epocas):
        modelo.train()
        acumulado, vistos = 0.0, 0
        for lote in lotes(indices_entrenamiento, tamano_lote, mezclar=True, generador=generador):
            logits, _ = modelo(X[lote].to(equipo), mascara[lote].to(equipo))
            valor = perdida_fn(logits, y[lote].to(equipo))

            optimizador.zero_grad(set_to_none=True)
            valor.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), recorte)
            optimizador.step()

            acumulado += float(valor.detach()) * len(lote)
            vistos += len(lote)

        pr_auc = float(average_precision_score(
            y_seguimiento, _logits(modelo, X, mascara, indices_seguimiento, equipo)
        ))
        historial.perdida.append(acumulado / vistos)
        historial.pr_auc_validacion.append(pr_auc)

        if pr_auc > mejor_pr_auc + 1e-5:
            mejor_pr_auc, sin_mejora = pr_auc, 0
            historial.mejor_epoca = epoca
            mejor_estado = copy.deepcopy(modelo.state_dict())
        else:
            sin_mejora += 1

        if verboso:
            marca = "  <- mejor" if sin_mejora == 0 else ""
            print(f"  epoca {epoca + 1:>2}  perdida {acumulado / vistos:.5f}"
                  f"  PR-AUC validacion {pr_auc:.4f}{marca}", flush=True)

        if sin_mejora >= paciencia:
            if verboso:
                print(f"  parada temprana tras {epoca + 1} epocas", flush=True)
            break

    modelo.load_state_dict(mejor_estado)
    historial.segundos = time.time() - inicio
    return modelo, historial


@torch.no_grad()
def predecir(
    modelo: Clasificador,
    tensores: Tensores,
    equipo: torch.device | None = None,
    tamano_lote: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve el logit y los pesos de atención de todas las secuencias."""
    equipo = equipo or dispositivo_disponible()
    modelo = modelo.to(equipo).eval()
    X = torch.from_numpy(tensores.X)
    mascara = torch.from_numpy(tensores.mascara)

    logits = np.zeros(len(tensores.y), dtype=np.float32)
    atencion = np.zeros(tensores.mascara.shape, dtype=np.float32)
    for lote in lotes(torch.arange(len(tensores.y)), tamano_lote):
        valores, pesos = modelo(X[lote].to(equipo), mascara[lote].to(equipo))
        logits[lote] = valores.cpu().numpy()
        atencion[lote] = pesos.cpu().numpy()
    return logits, atencion


def ajustar_fusion(error_a: np.ndarray, logit_b: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Combina las señales de ambas etapas con una regresión logística.

    Se elige un modelo lineal de dos entradas y no algo más expresivo porque una
    alerta de cumplimiento debe poder explicarse. Con dos coeficientes es
    posible decir cuánto aportó cada etapa a cada decisión.
    """
    entradas = np.column_stack([_estandarizar(error_a), logit_b])
    return LogisticRegression(max_iter=1000, class_weight="balanced").fit(entradas, y)


def aplicar_fusion(
    modelo: LogisticRegression, error_a: np.ndarray, logit_b: np.ndarray,
    referencia: np.ndarray | None = None,
) -> np.ndarray:
    """Probabilidad combinada de ambas etapas."""
    base = referencia if referencia is not None else error_a
    entradas = np.column_stack([_estandarizar(error_a, base), logit_b])
    return modelo.predict_proba(entradas)[:, 1]


def _estandarizar(valores: np.ndarray, referencia: np.ndarray | None = None) -> np.ndarray:
    """Lleva el error de reconstrucción a media cero y desviación uno."""
    base = referencia if referencia is not None else valores
    return (valores - base.mean()) / (base.std() + 1e-8)
