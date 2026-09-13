"""Comprobación de si PaySim admite secuencias por remitente.

PaySim es el conjunto más citado en detección de fraude financiero, pero antes
de usarlo hay que verificar que cada cuenta emisora reaparezca lo suficiente
como para formar una secuencia. Este módulo hace esa medición.

Uso:
    from data_pipeline.sourcing.validators import paysim

    conteos = paysim.conteos_por_cuenta(ruta)
    tabla = paysim.resumen(conteos, filas)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

COLUMNAS = ("nameOrig", "nameDest")
DESPLAZAMIENTO_COMERCIO = 10**12


def _a_entero(columna: pa.Array) -> np.ndarray:
    """Convierte `C1231006815` o `M1979787155` en un entero único por cuenta."""
    if isinstance(columna, pa.ChunkedArray):
        columna = columna.combine_chunks()
    numero = pc.cast(pc.utf8_slice_codeunits(columna, 1), pa.int64()).to_numpy(zero_copy_only=False)
    comercio = pc.equal(pc.utf8_slice_codeunits(columna, 0, 1), "M").to_numpy(zero_copy_only=False)
    return numero + comercio.astype(np.int64) * DESPLAZAMIENTO_COMERCIO


def conteos_por_cuenta(ruta: str | Path) -> tuple[dict[str, np.ndarray], int]:
    """Transacciones por cuenta, como emisora y como receptora.

    Devuelve un arreglo de conteos por cada columna, junto con el total de
    transacciones leídas.
    """
    lector = pv.open_csv(
        Path(ruta),
        read_options=pv.ReadOptions(block_size=1 << 21),
        convert_options=pv.ConvertOptions(include_columns=list(COLUMNAS)),
    )
    piezas: dict[str, list[np.ndarray]] = {columna: [] for columna in COLUMNAS}
    filas = 0
    for lote in lector:
        for columna in COLUMNAS:
            piezas[columna].append(_a_entero(lote[columna]))
        filas += lote.num_rows

    pa.default_memory_pool().release_unused()
    conteos = {}
    for columna in COLUMNAS:
        codigos = pd.factorize(np.concatenate(piezas.pop(columna)), sort=False)[0]
        conteos[columna] = np.bincount(codigos)
    return conteos, filas


def resumen(conteos: dict[str, np.ndarray], filas: int) -> pd.DataFrame:
    """Tabla comparativa entre el papel de emisor y el de receptor."""
    etiquetas = {"nameOrig": "como emisor (nameOrig)", "nameDest": "como receptor (nameDest)"}
    return pd.DataFrame(
        [
            {
                "cuentas unicas": int(valores.size),
                "transacciones por cuenta (media)": float(filas / valores.size),
                "maximo por cuenta": int(valores.max()),
                "con mas de una transaccion": float((valores > 1).mean()),
            }
            for valores in (conteos[columna] for columna in COLUMNAS)
        ],
        index=[etiquetas[columna] for columna in COLUMNAS],
    )
