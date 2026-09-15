"""Explicación en lenguaje natural de una alerta, generada con plantillas.

Una alerta de cumplimiento tiene que ser reproducible y auditable, es decir,
dos consultas sobre el mismo remitente deben producir exactamente el mismo
texto, y cada afirmación debe poder rastrearse hasta una transacción concreta.

Uso:
    from inference.utils.explicacion import contribucion, explicar

    pesos = contribucion(atencion, error_por_paso, mascara)
    texto = explicar(resumen, contexto, pesos)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRANSACCIONES_CITADAS = 3


@dataclass
class Resumen:
    """Lo que el sistema concluyó sobre un remitente."""

    identificador: str
    probabilidad: float
    error: float
    umbral_probabilidad: float
    umbral_error: float

    @property
    def alerta(self) -> bool:
        """Indica si el remitente supera el umbral de la probabilidad combinada."""
        return self.probabilidad >= self.umbral_probabilidad

    @property
    def anomalo(self) -> bool:
        """Indica si el error de reconstrucción supera su propio umbral."""
        return self.error >= self.umbral_error


def _normalizar(valores: np.ndarray) -> np.ndarray:
    """Lleva un vector al rango cero a uno, con ceros si es constante."""
    rango = float(valores.max() - valores.min())
    if rango <= 0:
        return np.zeros_like(valores)
    return (valores - valores.min()) / rango


def contribucion(
    atencion: np.ndarray, error_por_paso: np.ndarray, mascara: np.ndarray
) -> pd.DataFrame:
    """Contribución de cada paso a la alerta, por etapa y combinada.

    La columna `atencion` es el peso que la etapa B dió a cada transacción, y
    `anomalia` es el error de reconstrucción de la etapa A. Ambas se llevan al
    rango cero a uno dentro de la secuencia para poder promediarlas.
    """
    pasos = int(mascara.sum())
    tabla = pd.DataFrame({
        "paso": np.arange(pasos),
        "atencion": atencion[:pasos],
        "anomalia": error_por_paso[:pasos],
    })
    tabla["atencion_relativa"] = _normalizar(tabla["atencion"].to_numpy())
    tabla["anomalia_relativa"] = _normalizar(tabla["anomalia"].to_numpy())
    tabla["contribucion"] = (tabla["atencion_relativa"] + tabla["anomalia_relativa"]) / 2
    return tabla


def _describir_transaccion(fila: pd.Series) -> str:
    """Frase corta que describe una transacción citada."""
    partes = [f"{fila['fecha']:%d de septiembre a las %H:%M}",
              f"por {fila['monto_usd']:,.0f} dolares",
              f"vía {fila['formato']}"]
    marcas = []
    if fila.get("destino_nuevo"):
        marcas.append("a un destino nunca antes usado")
    if fila.get("cruce_moneda"):
        marcas.append("con cambio de moneda")
    if fila.get("cerca_umbral"):
        marcas.append("con un monto apenas por debajo del umbral de reporte")
    if marcas:
        partes.append(", ".join(marcas))
    return " ".join(partes)


def _senal_dominante(fila: pd.Series) -> str:
    """Indica cuál de las dos etapas hizo destacar a la transaccioón."""
    if fila["atencion_relativa"] >= fila["anomalia_relativa"]:
        return f"por el peso de atención del clasificador, que le asigna el {fila['atencion']:.0%}"
    return "por lo difícil que resulta de reconstruir para el modelo de normalidad"


def explicar(resumen: Resumen, contexto: pd.DataFrame, pesos: pd.DataFrame) -> str:
    """Redacta el parrafo que acompaña a la alerta.

    `contexto` debe traer una fila por paso con las columnas `fecha`,
    `monto_usd`, `formato`, `destino`, y los indicadores `destino_nuevo`,
    `cruce_moneda` y `cerca_umbral`.
    """
    tabla = contexto.reset_index(drop=True).join(pesos.set_index("paso"), on="paso")
    citadas = tabla.nlargest(TRANSACCIONES_CITADAS, "contribucion").sort_values("paso")

    veredicto = (
        f"El remitente {resumen.identificador} recibe una probabilidad de lavado de "
        f"{resumen.probabilidad:.1%}, {'por encima' if resumen.alerta else 'por debajo'} del "
        f"umbral operativo de {resumen.umbral_probabilidad:.1%}."
    )
    normalidad = (
        f"Su error de reconstrucción es de {resumen.error:.3f}, "
        f"{'superior' if resumen.anomalo else 'inferior'} al umbral de {resumen.umbral_error:.3f}, "
        f"de modo que su comportamiento "
        f"{'no se parece' if resumen.anomalo else 'se parece'} al de un remitente normal."
    )

    detalle = " ".join(
        f"La transacción {int(fila['paso']) + 1} de la secuencia, {_describir_transaccion(fila)}, "
        f"destaca {_senal_dominante(fila)}."
        for _, fila in citadas.iterrows()
    )

    destinos = tabla["destino"].nunique()
    patrones = []
    if destinos >= max(3, int(0.6 * len(tabla))):
        patrones.append(
            f"envía a {destinos} destinos distintos en {len(tabla)} transacciones, "
            "un patrón de dispersión"
        )
    if tabla["destino_nuevo"].sum() >= 3:
        patrones.append(f"dirige {int(tabla['destino_nuevo'].sum())} operaciones a destinos nuevos")
    if tabla["cerca_umbral"].sum() >= 2:
        patrones.append(
            f"coloca {int(tabla['cerca_umbral'].sum())} montos justo debajo del umbral de reporte"
        )
    if tabla["cruce_moneda"].sum() >= 2:
        patrones.append(f"cambia de moneda en {int(tabla['cruce_moneda'].sum())} operaciones")

    contexto_general = (
        f"En el conjunto de la ventana, el remitente {'; '.join(patrones)}."
        if patrones
        else "En el conjunto de la ventana no se observan patrones agregados destacables."
    )

    cierre = (
        "Se recomienda abrir una revisión manual y validar el origen de los fondos."
        if resumen.alerta
        else "No se recomienda abrir una revisión con la información disponible."
    )

    return " ".join([veredicto, normalidad, detalle, contexto_general, cierre])
