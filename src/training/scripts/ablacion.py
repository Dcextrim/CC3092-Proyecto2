"""Experimento de ablación sobre la arquitectura de dos etapas.

Compara el clasificador entrenado desde cero contra las dos formas de reutilizar
el codificador de la etapa A, y mide cuánto agrega combinar ambas señales. Cada
configuración se repite con varias semillas, porque con 292 positivos en prueba
una sola corrida no distingue una mejora real del ruido.

Uso:
    from training.scripts.ablacion import ejecutar_ablacion

    tabla, artefactos = ejecutar_ablacion(datos, modelo_a)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from models.metrics.deteccion import evaluar, umbral_por_presupuesto
from training.scripts.entrenar_etapa_a import Puntajes
from training.scripts.entrenar_etapa_b import (
    ajustar_fusion,
    aplicar_fusion,
    entrenar_clasificador,
    predecir,
)
from utils.config import obtener


@dataclass(frozen=True)
class Configuracion:
    """Una fila del experimento de ablación."""

    clave: str
    nombre: str
    estrategia: str
    perdida: str
    fusion: bool = False


CONFIGURACIONES = (
    Configuracion("B1", "Supervisado desde cero", "desde_cero", "focal"),
    Configuracion("B2", "Codificador congelado", "extraccion", "focal"),
    Configuracion("B3", "Ajuste parcial, learning rate diferencial", "ajuste_parcial", "focal"),
    Configuracion("B4", "Ajuste parcial mas fusion con etapa A", "ajuste_parcial", "focal", True),
    Configuracion("B5", "Ajuste parcial con entropia ponderada", "ajuste_parcial", "ponderada"),
    Configuracion("B6", "Pesos de la etapa A con learning rate uniforme", "ajuste_uniforme", "focal"),
)

ENTRENADAS = tuple(c for c in CONFIGURACIONES if not c.fusion)


def evaluar_etapa_a(datos, puntajes: Puntajes, presupuesto: float) -> dict[str, float]:
    """Métricas de la etapa A sola, usando el error de reconstrucción como puntaje."""
    y = datos.tensores.y
    validacion, prueba = datos.particion.validacion, datos.particion.prueba
    umbral = umbral_por_presupuesto(puntajes.error[validacion], presupuesto)
    return evaluar(y[prueba], puntajes.error[prueba], presupuesto, umbral=umbral)


def ejecutar_ablacion(
    datos,
    modelo_a,
    puntajes_a: Puntajes,
    semillas: tuple[int, ...] | None = None,
    equipo: torch.device | None = None,
    verboso: bool = True,
    limite_entrenamiento: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Entrena cada configuración con varias semillas y mide sobre el conjunto de prueba.

    El umbral operativo se fija siempre sobre validación y luego se aplica a
    prueba, de modo que ninguna métrica reportada usa información del conjunto
    de prueba para calibrarse.
    """
    semillas = semillas or tuple(obtener("entrenamiento.semillas", [0, 1, 2]))
    presupuesto = obtener("entrenamiento.presupuesto_alertas", 0.01)
    pesos = {clave: valor.cpu() for clave, valor in modelo_a.codificador.state_dict().items()}

    y = datos.tensores.y
    validacion, prueba = datos.particion.validacion, datos.particion.prueba

    filas: list[dict[str, object]] = []
    artefactos: dict[str, object] = {"pesos_etapa_a": pesos}

    for configuracion in ENTRENADAS:
        for semilla in semillas:
            if verboso:
                print(f"\n{configuracion.clave} {configuracion.nombre}  semilla {semilla}", flush=True)
            modelo, historial = entrenar_clasificador(
                datos, configuracion.estrategia, pesos, configuracion.perdida,
                semilla=semilla, equipo=equipo, verboso=verboso,
                limite_entrenamiento=limite_entrenamiento,
            )
            logit, atencion = predecir(modelo, datos.tensores, equipo=equipo)

            umbral = umbral_por_presupuesto(logit[validacion], presupuesto)
            metricas = evaluar(y[prueba], logit[prueba], presupuesto, umbral=umbral)
            filas.append({
                "clave": configuracion.clave, "configuracion": configuracion.nombre,
                "semilla": semilla, "epocas": len(historial.perdida),
                "segundos": round(historial.segundos, 1), **metricas,
            })

            if configuracion.fusion is False and configuracion.clave == "B3":
                fusion = ajustar_fusion(puntajes_a.error[validacion], logit[validacion], y[validacion])
                probabilidad = aplicar_fusion(
                    fusion, puntajes_a.error, logit, referencia=puntajes_a.error[validacion]
                )
                umbral_fusion = umbral_por_presupuesto(probabilidad[validacion], presupuesto)
                combinada = evaluar(y[prueba], probabilidad[prueba], presupuesto, umbral=umbral_fusion)
                cuatro = next(c for c in CONFIGURACIONES if c.clave == "B4")
                filas.append({
                    "clave": cuatro.clave, "configuracion": cuatro.nombre,
                    "semilla": semilla, "epocas": len(historial.perdida),
                    "segundos": round(historial.segundos, 1), **combinada,
                })
                if semilla == semillas[0]:
                    artefactos.update({
                        "modelo_b": modelo, "fusion": fusion, "logit_b": logit,
                        "atencion_b": atencion, "probabilidad": probabilidad,
                        "umbral_b": umbral, "umbral_fusion": umbral_fusion,
                        "referencia_error": puntajes_a.error[validacion],
                    })

    return pd.DataFrame(filas), artefactos


def resumir(tabla: pd.DataFrame, metricas: tuple[str, ...] = ("pr_auc", "recall", "precision", "f2", "roc_auc")) -> pd.DataFrame:
    """Promedio y desviación de cada métrica por configuración."""
    agregado = tabla.groupby(["clave", "configuracion"], sort=True)[list(metricas)].agg(["mean", "std"])
    salida = pd.DataFrame(index=agregado.index)
    for metrica in metricas:
        salida[metrica] = [
            f"{media:.4f} ± {0.0 if np.isnan(desviacion) else desviacion:.4f}"
            for media, desviacion in zip(agregado[(metrica, "mean")], agregado[(metrica, "std")])
        ]
    return salida.reset_index()


# Las tres configuraciones que la curva de escasez compara. B1 y B3 son la linea
# base (extraida de la semana 9), y B6 comparte con B3 la inicialización pero
# con B1 el learning rate, de modo que la terna separa el efecto de partir de la
# etapa A del efecto de moverse más despacio.
ESCASEZ = ("B1", "B3", "B6")


def ejecutar_curva_de_escasez(
    datos,
    modelo_a,
    fracciones: tuple[float, ...] | None = None,
    semillas: tuple[int, ...] | None = None,
    equipo: torch.device | None = None,
    verboso: bool = False,
) -> pd.DataFrame:
    """Mide si la transferencia aporta cuando hay pocos ejemplos etiquetados.

    Reentrena la linea base y el ajuste parcial con fracciones crecientes de los
    remitentes positivos de entrenamiento. La semana 9 sostiene que la
    transferencia rinde sobre todo con conjuntos objetivo pequeños, y esta curva
    pone esa afirmación a prueba con los datos del proyecto.
    """
    fracciones = fracciones or tuple(obtener("entrenamiento.fracciones_de_escasez", [0.1, 0.25, 0.5]))
    semillas = semillas or tuple(obtener("entrenamiento.semillas_de_escasez", [0, 1]))
    presupuesto = obtener("entrenamiento.presupuesto_alertas", 0.01)
    pesos = {clave: valor.cpu() for clave, valor in modelo_a.codificador.state_dict().items()}

    y = datos.tensores.y
    validacion, prueba = datos.particion.validacion, datos.particion.prueba
    positivos_totales = int(y[datos.particion.entrenamiento].sum())
    elegidas = [c for c in ENTRENADAS if c.clave in ESCASEZ]

    filas: list[dict[str, object]] = []
    for fraccion in fracciones:
        for configuracion in elegidas:
            for semilla in semillas:
                print(f"  fraccion {fraccion:.0%}  {configuracion.clave}  semilla {semilla}", flush=True)
                modelo, _ = entrenar_clasificador(
                    datos, configuracion.estrategia, pesos, configuracion.perdida,
                    semilla=semilla, equipo=equipo, verboso=verboso,
                    fraccion_positivos=fraccion,
                )
                logit, _ = predecir(modelo, datos.tensores, equipo=equipo)
                umbral = umbral_por_presupuesto(logit[validacion], presupuesto)
                filas.append({
                    "fraccion": fraccion,
                    "positivos": int(round(fraccion * positivos_totales)),
                    "clave": configuracion.clave,
                    "configuracion": configuracion.nombre,
                    "semilla": semilla,
                    **evaluar(y[prueba], logit[prueba], presupuesto, umbral=umbral),
                })
    return pd.DataFrame(filas)


def unir_escasez(
    curva: pd.DataFrame, tabla_ablacion: pd.DataFrame, positivos_totales: int
) -> pd.DataFrame:
    """Agrega el punto de 100% de etiquetas, tomado de la ablación principal.

    Ese punto se reutiliza en lugar de reentrenarlo, porque es exactamente la
    misma configuración que ya midió la tabla principal, solo que con tres
    semillas en vez de dos.
    """
    completo = tabla_ablacion[tabla_ablacion["clave"].isin(ESCASEZ)].copy()
    completo["fraccion"] = 1.0
    completo["positivos"] = positivos_totales
    return pd.concat([curva, completo], ignore_index=True)
