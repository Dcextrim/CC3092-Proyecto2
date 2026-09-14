"""Métricas de detección para un problema con desbalance extremo.

Con 72 remitentes normales por cada sospechoso, la exactitud no informa nada,
porque predecir siempre la clase mayoritaria acierta el 98.6% de las veces. Las
métricas que sí informan son las que miran la clase positiva y las que se
expresan en términos de la carga de trabajo del equipo que revisa las alertas.

Uso:
    from models.metrics.deteccion import evaluar, umbral_por_presupuesto

    resultados = evaluar(y_prueba, puntaje, presupuesto=0.01)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def umbral_por_presupuesto(puntaje: np.ndarray, presupuesto: float) -> float:
    """Umbral que marca como alerta la fracción `presupuesto` de mayor puntaje.

    Traduce la capacidad real de revisión de un equipo de cumplimiento en un
    punto de corte, en lugar de fijarlo por conveniencia estadística.
    """
    return float(np.quantile(puntaje, 1.0 - presupuesto))


def f_beta(precision: np.ndarray, recall: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """F-beta a partir de precision y recall, con cero donde ambos son cero."""
    beta2 = beta**2
    denominador = beta2 * precision + recall
    return np.divide(
        (1 + beta2) * precision * recall, denominador,
        out=np.zeros_like(denominador), where=denominador > 0,
    )


def umbral_por_f_beta(
    y: np.ndarray, puntaje: np.ndarray, beta: float = 2.0
) -> tuple[float, float]:
    """Umbral que maximiza F-beta sobre validación, y el valor alcanzado.

    Con beta igual a 2 el recall pesa cuatro veces más que la precision, que es
    la preferencia razonable cuando el costo de dejar pasar un caso de lavado
    supera al de revisar una alerta de más.
    """
    precision, recall, umbrales = precision_recall_curve(y, puntaje)
    valores = f_beta(precision[:-1], recall[:-1], beta)
    mejor = int(np.argmax(valores))
    return float(umbrales[mejor]), float(valores[mejor])


def evaluar(
    y: np.ndarray, puntaje: np.ndarray, presupuesto: float = 0.01, umbral: float | None = None
) -> dict[str, float]:
    """Resume el desempeño de un puntaje continuo sobre la clase positiva.

    `pr_auc` es la métrica principal porque solo mira la clase positiva y no se
    deja inflar por los verdaderos negativos. Las métricas con presupuesto
    describen lo que ocurre cuando solo se revisa el 1% de los remitentes.
    """
    corte = umbral if umbral is not None else umbral_por_presupuesto(puntaje, presupuesto)
    alerta = puntaje >= corte
    positivos = y == 1

    detectados = int((alerta & positivos).sum())
    alertas = int(alerta.sum())
    precision = detectados / alertas if alertas else 0.0
    recall = detectados / int(positivos.sum()) if positivos.any() else 0.0

    return {
        "pr_auc": float(average_precision_score(y, puntaje)),
        "roc_auc": float(roc_auc_score(y, puntaje)),
        "recall": recall,
        "precision": precision,
        "f2": float(f_beta(np.array([precision]), np.array([recall]), 2.0)[0]),
        "alertas": alertas,
        "tasa_de_alerta": alertas / len(y),
        "umbral": corte,
    }


def resumen_multiple(resultados: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Promedio y desviación de cada métrica sobre varias semillas."""
    claves = resultados[0].keys()
    return {
        clave: (
            float(np.mean([r[clave] for r in resultados])),
            float(np.std([r[clave] for r in resultados])),
        )
        for clave in claves
    }
