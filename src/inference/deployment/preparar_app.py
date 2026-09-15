"""Artefactos que consume el MVP desplegado.

La aplicación no reconstruye el pipeline ni descarga el archivo original. Recibe
una muestra del conjunto de prueba ya tensorizada, el contexto legible de cada
transacción y los umbrales calibrados, y hace inferencia real sobre el modelo
ONNX.

La muestra conserva todos los remitentes positivos de prueba y completa con
negativos tomados al azar, de modo que el evaluador siempre encuentra casos de
ambas clases sin que el repositorio cargue con decenas de megabytes.

Uso:
    from inference.deployment.preparar_app import preparar

    preparar(datos, umbrales, destino)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_pipeline.preprocessing.features import VARIABLES

NEGATIVOS_EN_MUESTRA = 708


def seleccionar_muestra(
    y: np.ndarray, prueba: np.ndarray, negativos: int = NEGATIVOS_EN_MUESTRA, semilla: int = 42
) -> np.ndarray:
    """Todos los positivos de prueba más una muestra fija de negativos."""
    positivos = prueba[y[prueba] == 1]
    candidatos = prueba[y[prueba] == 0]
    elegidos = np.random.default_rng(semilla).choice(
        candidatos, size=min(negativos, len(candidatos)), replace=False
    )
    return np.sort(np.concatenate([positivos, elegidos]))


def preparar(
    datos,
    umbrales: dict[str, float],
    destino: Path,
    semilla: int = 42,
) -> dict[str, object]:
    """Escribe la muestra, el contexto y los metadatos en `destino`.

    Devuelve un resumen con el tamaño de la muestra y de los archivos escritos.
    """
    destino.mkdir(parents=True, exist_ok=True)
    tensores = datos.tensores
    indices = seleccionar_muestra(tensores.y, datos.particion.prueba, semilla=semilla)

    np.savez_compressed(
        destino / "muestra.npz",
        X=tensores.X[indices],
        mascara=tensores.mascara[indices],
        y=tensores.y[indices],
        longitud=tensores.longitud[indices],
        identificador=tensores.identificador[indices].astype("U16"),
    )

    contexto = tensores.contexto
    contexto = contexto[contexto["emisor"].isin(indices)].copy()
    posicion = {emisor: orden for orden, emisor in enumerate(indices)}
    contexto["remitente"] = contexto["emisor"].map(posicion)
    contexto = contexto.drop(columns=["emisor"]).sort_values(["remitente", "paso"])
    contexto.to_parquet(destino / "contexto.parquet", index=False)

    (destino / "meta.json").write_text(
        json.dumps({
            "variables": VARIABLES,
            "catalogos": datos.catalogos,
            "umbrales": umbrales,
            "remitentes": int(len(indices)),
            "positivos": int(tensores.y[indices].sum()),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "remitentes": int(len(indices)),
        "positivos": int(tensores.y[indices].sum()),
        "transacciones": int(len(contexto)),
        "megabytes": round(
            sum(archivo.stat().st_size for archivo in destino.iterdir()) / 1e6, 2
        ),
    }
