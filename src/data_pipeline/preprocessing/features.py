"""Lectura del CSV de IBM AML y variables por transacción.

Uso:
    from data_pipeline.preprocessing import features

    crudo, catalogos = features.cargar_transacciones()
    ventana = features.construir_ventanas(crudo)
    tabla = features.construir_variables(ventana, catalogos)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

from utils.config import obtener
from utils.rutas import DATA_RAW

COLUMNAS = [
    "fecha", "banco_emisor", "cuenta_emisor", "banco_receptor", "cuenta_receptor",
    "monto_recibido", "moneda_recibida", "monto_pagado", "moneda_pagada",
    "formato_pago", "es_lavado",
]

TIPOS_LECTURA = {
    "fecha": pa.string(),
    "banco_emisor": pa.string(), "cuenta_emisor": pa.string(),
    "banco_receptor": pa.string(), "cuenta_receptor": pa.string(),
    "monto_recibido": pa.float32(), "monto_pagado": pa.float32(),
    "moneda_recibida": pa.string(), "moneda_pagada": pa.string(),
    "formato_pago": pa.string(), "es_lavado": pa.int8(),
}

FORMATOS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]

CONTEXTO = [
    "emisor", "banco_emisor", "cuenta_emisor", "banco_receptor", "cuenta_receptor",
    "fecha", "formato_pago", "moneda_pagada", "es_lavado",
]

VARIABLES = [
    "log_monto",
    "log_delta_t",
    "hora_sin",
    "hora_cos",
    "dia_sin",
    "dia_cos",
    *[f"formato_{f.lower().replace(' ', '_')}" for f in FORMATOS],
    "cruce_banco",
    "cruce_moneda",
    "auto_transferencia",
    "destino_nuevo",
    "log_repeticiones_destino",
    "desvio_monto",
    "cerca_umbral",
]

_LARGO_CUENTA = 9
_PESOS_HEX = (16 ** np.arange(_LARGO_CUENTA - 1, -1, -1)).astype(np.int64)
_BITS_CUENTA = 36


def _hex_a_entero(columna: pa.Array) -> np.ndarray:
    """Convierte cuentas hexadecimales de nueve caracteres a enteros."""
    columna = columna.combine_chunks() if isinstance(columna, pa.ChunkedArray) else columna
    crudo = np.frombuffer(
        columna.buffers()[2], dtype=np.uint8, count=len(columna) * _LARGO_CUENTA
    ).reshape(-1, _LARGO_CUENTA)
    salida = np.zeros(len(columna), dtype=np.int64)
    for posicion in range(_LARGO_CUENTA):
        digito = crudo[:, posicion].astype(np.int16) - 48
        np.subtract(digito, 7, out=digito, where=digito > 9)
        salida += digito.astype(np.int64) * _PESOS_HEX[posicion]
    return salida


def _codificar(columna: pa.Array, catalogo: dict[str, int]) -> np.ndarray:
    """Asigna un código entero estable a cada categoría, compartido entre lotes."""
    diccionario = columna.dictionary_encode()
    if isinstance(diccionario, pa.ChunkedArray):
        diccionario = diccionario.combine_chunks()
    traduccion = np.array(
        [catalogo.setdefault(valor, len(catalogo)) for valor in diccionario.dictionary.to_pylist()],
        dtype=np.int8,
    )
    return traduccion[diccionario.indices.to_numpy(zero_copy_only=False)]


def cargar_transacciones(ruta: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Lee el CSV crudo por lotes y devuelve la tabla compacta y sus catalogos."""
    destino = Path(ruta) if ruta else DATA_RAW / obtener("datos.archivo", "HI-Small_Trans.csv")
    lector = pv.open_csv(
        destino,
        read_options=pv.ReadOptions(column_names=COLUMNAS, skip_rows=1, block_size=1 << 21),
        convert_options=pv.ConvertOptions(column_types=TIPOS_LECTURA),
    )

    piezas: dict[str, list[np.ndarray]] = {
        clave: [] for clave in (
            "cuenta_emisor", "cuenta_receptor", "banco_emisor", "banco_receptor",
            "fecha", "monto_pagado", "monto_recibido",
            "moneda_pagada", "moneda_recibida", "formato_pago", "es_lavado",
        )
    }
    monedas: dict[str, int] = {}
    formatos: dict[str, int] = {}

    for lote in lector:
        piezas["cuenta_emisor"].append(_hex_a_entero(lote["cuenta_emisor"]))
        piezas["cuenta_receptor"].append(_hex_a_entero(lote["cuenta_receptor"]))
        piezas["banco_emisor"].append(
            pc.cast(lote["banco_emisor"], pa.int32()).to_numpy(zero_copy_only=False))
        piezas["banco_receptor"].append(
            pc.cast(lote["banco_receptor"], pa.int32()).to_numpy(zero_copy_only=False))
        marca = pc.strptime(lote["fecha"], format="%Y/%m/%d %H:%M", unit="s")
        piezas["fecha"].append(marca.cast(pa.int64()).to_numpy(zero_copy_only=False))
        piezas["monto_pagado"].append(lote["monto_pagado"].to_numpy(zero_copy_only=False))
        piezas["monto_recibido"].append(lote["monto_recibido"].to_numpy(zero_copy_only=False))
        piezas["moneda_pagada"].append(_codificar(lote["moneda_pagada"], monedas))
        piezas["moneda_recibida"].append(_codificar(lote["moneda_recibida"], monedas))
        piezas["formato_pago"].append(_codificar(lote["formato_pago"], formatos))
        piezas["es_lavado"].append(lote["es_lavado"].to_numpy(zero_copy_only=False))

    tabla = pd.DataFrame({clave: np.concatenate(valor) for clave, valor in piezas.items()})
    piezas.clear()
    # El lector de Arrow retiene los bloques liberados en su propio pool, que en
    # este punto ya suman varios cientos de megabytes sin uso.
    pa.default_memory_pool().release_unused()

    catalogos = {
        "monedas": [nombre for nombre, _ in sorted(monedas.items(), key=lambda par: par[1])],
        "formatos": [nombre for nombre, _ in sorted(formatos.items(), key=lambda par: par[1])],
    }
    return tabla, catalogos


def identificador_emisor(banco: np.ndarray, cuenta: np.ndarray) -> np.ndarray:
    """Reconstruye el identificador legible del remitente, por ejemplo `010_8000EBD30`."""
    return np.array(
        [f"{int(b):03d}_{int(c):09X}" for b, c in zip(banco, cuenta)], dtype=object
    )


def tasas_a_usd(crudo: pd.DataFrame, catalogos: dict[str, list[str]]) -> dict[str, float]:
    """Deriva la tasa de cambio de cada moneda a dólares desde el propio dataset.

    Los montos vienen en quince monedas distintas y no son comparables entre sí.
    Las transacciones que cambian de moneda revelan la tasa implícita, y se toma
    la mediana de aquellas que liquidan en dólares.
    """
    monedas = catalogos["monedas"]
    codigo_usd = monedas.index("US Dollar")
    cambio = crudo["moneda_pagada"].to_numpy() != crudo["moneda_recibida"].to_numpy()
    hacia_usd = cambio & (crudo["moneda_recibida"].to_numpy() == codigo_usd)

    implicita = (
        crudo.loc[hacia_usd, "monto_recibido"].to_numpy()
        / crudo.loc[hacia_usd, "monto_pagado"].to_numpy()
    )
    origen = crudo.loc[hacia_usd, "moneda_pagada"].to_numpy()
    tasas = {"US Dollar": 1.0}
    for codigo in np.unique(origen):
        tasas[monedas[codigo]] = float(np.median(implicita[origen == codigo]))

    faltantes = set(monedas) - set(tasas)
    if faltantes:
        raise ValueError(f"Sin tasa implicita hacia dolares para: {sorted(faltantes)}")
    return tasas


def _limites(codigo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve conteo, inicio y posición dentro de cada grupo contiguo ya ordenado."""
    conteo = np.bincount(codigo)
    inicio = np.cumsum(conteo) - conteo
    posicion = np.arange(len(codigo), dtype=np.int64) - np.repeat(inicio, conteo)
    return conteo, inicio, posicion


def _desplazar(valores: np.ndarray, posicion: np.ndarray, relleno: float) -> np.ndarray:
    """Valor anterior dentro del grupo, con `relleno` en el primer paso."""
    anterior = np.empty_like(valores)
    anterior[0] = relleno
    anterior[1:] = valores[:-1]
    anterior[posicion == 0] = relleno
    return anterior


def _suma_previa(valores: np.ndarray, conteo: np.ndarray, inicio: np.ndarray) -> np.ndarray:
    """Suma acumulada dentro del grupo, sin incluir el valor actual."""
    acumulado = np.cumsum(valores)
    base = np.zeros(len(conteo), dtype=acumulado.dtype)
    base[1:] = acumulado[inicio[1:] - 1]
    return acumulado - np.repeat(base, conteo) - valores


def construir_ventanas(
    crudo: pd.DataFrame,
    minimo_transacciones: int | None = None,
    longitud_maxima: int | None = None,
    liberar: bool = True,
) -> pd.DataFrame:
    """Conserva a los remitentes con historial suficiente y recorta su ventana.

    Descarta a quien tenga menos de `minimo_transacciones` salientes, ordena por
    remitente y fecha, y se queda con las ultimas `longitud_maxima` transacciones
    de cada uno. La ventana resultante es la unidad de observacion del sistema.

    Con `liberar` activo, cada columna se descarta de `crudo` en cuanto se copia,
    lo que evita mantener las dos tablas completas a la vez y deja inutilizable
    la tabla recibida.
    """
    minimo = minimo_transacciones or obtener("secuencias.minimo_transacciones", 4)
    largo = longitud_maxima or obtener("secuencias.longitud_maxima", 32)

    clave = (
        crudo["banco_emisor"].to_numpy().astype(np.int64) << _BITS_CUENTA
    ) + crudo["cuenta_emisor"].to_numpy()
    codigo = pd.factorize(clave, sort=False)[0]
    del clave

    conteo = np.bincount(codigo)
    filas = np.flatnonzero(conteo[codigo] >= minimo)
    codigo = pd.factorize(codigo[filas], sort=False)[0].astype(np.int32)

    orden = np.lexsort((crudo["fecha"].to_numpy()[filas], codigo))
    filas = filas[orden]
    codigo = codigo[orden]
    del orden

    conteo, _, posicion = _limites(codigo)
    recientes = (np.repeat(conteo, conteo) - posicion) <= largo

    elegidas = filas[recientes]
    emisor = codigo[recientes]
    del filas, recientes, codigo, conteo, posicion

    if liberar:
        ventana = pd.DataFrame(
            {nombre: crudo.pop(nombre).to_numpy()[elegidas] for nombre in list(crudo.columns)}
        )
    else:
        ventana = crudo.take(elegidas).reset_index(drop=True)

    ventana["emisor"] = emisor
    return ventana


def construir_variables(
    ventana: pd.DataFrame,
    catalogos: dict[str, list[str]],
    tasas: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Calcula las variables de `VARIABLES` sobre una ventana ya ordenada.

    Espera la salida de `construir_ventanas`, es decir filas ordenadas por
    remitente y fecha con la columna `emisor`. Conviene pasar `tasas` estimadas
    sobre el archivo completo, porque la ventana contiene menos transacciones
    con cambio de moneda y daria estimaciones más ruidosas.
    """
    umbral = float(obtener("umbral_estructuracion.monto_usd", 10000))
    banda = float(obtener("umbral_estructuracion.fraccion_inferior", 0.75))
    tasas = tasas if tasas is not None else tasas_a_usd(ventana, catalogos)

    emisor = ventana["emisor"].to_numpy()
    conteo, inicio, posicion = _limites(emisor)
    filas = len(ventana)
    matriz = np.zeros((filas, len(VARIABLES)), dtype=np.float32)
    columna = {nombre: indice for indice, nombre in enumerate(VARIABLES)}

    factores = np.array([tasas[moneda] for moneda in catalogos["monedas"]], dtype=np.float64)
    monto_usd = ventana["monto_pagado"].to_numpy() * factores[ventana["moneda_pagada"].to_numpy()]
    log_monto = np.log1p(monto_usd)
    matriz[:, columna["log_monto"]] = log_monto

    segundos = ventana["fecha"].to_numpy().astype(np.float64)
    horas = (segundos - _desplazar(segundos, posicion, np.nan)) / 3600.0
    matriz[:, columna["log_delta_t"]] = np.nan_to_num(np.log1p(np.clip(horas, 0, None)))
    del horas

    fechas = pd.to_datetime(ventana["fecha"].to_numpy(), unit="s")
    hora = fechas.hour.to_numpy() + fechas.minute.to_numpy() / 60.0
    matriz[:, columna["hora_sin"]] = np.sin(2 * np.pi * hora / 24.0)
    matriz[:, columna["hora_cos"]] = np.cos(2 * np.pi * hora / 24.0)
    dia = fechas.dayofweek.to_numpy()
    matriz[:, columna["dia_sin"]] = np.sin(2 * np.pi * dia / 7.0)
    matriz[:, columna["dia_cos"]] = np.cos(2 * np.pi * dia / 7.0)
    del fechas, hora, dia

    formato = ventana["formato_pago"].to_numpy()
    for nombre in FORMATOS:
        clave = f"formato_{nombre.lower().replace(' ', '_')}"
        matriz[:, columna[clave]] = formato == catalogos["formatos"].index(nombre)

    matriz[:, columna["cruce_banco"]] = (
        ventana["banco_emisor"].to_numpy() != ventana["banco_receptor"].to_numpy()
    )
    matriz[:, columna["cruce_moneda"]] = (
        ventana["moneda_pagada"].to_numpy() != ventana["moneda_recibida"].to_numpy()
    )

    clave_receptor = (
        ventana["banco_receptor"].to_numpy().astype(np.int64) << _BITS_CUENTA
    ) + ventana["cuenta_receptor"].to_numpy()
    clave_emisor = (
        ventana["banco_emisor"].to_numpy().astype(np.int64) << _BITS_CUENTA
    ) + ventana["cuenta_emisor"].to_numpy()
    matriz[:, columna["auto_transferencia"]] = clave_emisor == clave_receptor
    del clave_emisor

    pareja = pd.factorize(
        emisor.astype(np.int64) * (int(clave_receptor.max()) + 1) + clave_receptor, sort=False
    )[0]
    del clave_receptor
    orden_pareja = np.argsort(pareja, kind="stable")
    conteo_pareja = np.bincount(pareja)
    inicio_pareja = np.cumsum(conteo_pareja) - conteo_pareja
    repeticiones = np.empty(filas, dtype=np.int64)
    repeticiones[orden_pareja] = np.arange(filas) - np.repeat(inicio_pareja, conteo_pareja)
    del pareja, orden_pareja, conteo_pareja, inicio_pareja
    matriz[:, columna["destino_nuevo"]] = repeticiones == 0
    matriz[:, columna["log_repeticiones_destino"]] = np.log1p(repeticiones)
    del repeticiones

    previa = _suma_previa(log_monto, conteo, inicio)
    with np.errstate(invalid="ignore", divide="ignore"):
        media_previa = np.where(posicion > 0, previa / np.maximum(posicion, 1), log_monto)
    matriz[:, columna["desvio_monto"]] = log_monto - media_previa
    del previa, media_previa, log_monto

    matriz[:, columna["cerca_umbral"]] = (monto_usd >= banda * umbral) & (monto_usd < umbral)

    tabla = pd.DataFrame(matriz, columns=VARIABLES)
    for nombre in CONTEXTO:
        tabla[nombre] = ventana[nombre].to_numpy() if nombre in ventana else None
    tabla["monto_usd"] = monto_usd.astype(np.float32)
    return tabla
