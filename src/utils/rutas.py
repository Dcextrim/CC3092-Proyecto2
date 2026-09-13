"""Rutas del proyecto y ayudas de lectura y escritura.

Contiene únicamente lo que se reutiliza entre notebooks y scripts: la raíz del
repositorio, las carpetas de datos y resultados, y dos ayudas para guardar y
cargar artefactos en `data/processed`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]

DATA_RAW = RAIZ / "data" / "raw"
DATA_INTERIM = RAIZ / "data" / "interim"
DATA_PROCESSED = RAIZ / "data" / "processed"

RESULTS_MODELS = RAIZ / "results" / "models"
RESULTS_PLOTS = RAIZ / "results" / "plots"
RESULTS_TABLES = RAIZ / "results" / "tables"

CONFIG = RAIZ / "config"


def asegurar_carpetas() -> None:
    """Crea las carpetas de datos y resultados si no existen."""
    for carpeta in (
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        RESULTS_MODELS,
        RESULTS_PLOTS,
        RESULTS_TABLES,
    ):
        carpeta.mkdir(parents=True, exist_ok=True)


def guardar_npz(nombre: str, **arreglos: np.ndarray) -> Path:
    """Guarda arreglos comprimidos en `data/processed/<nombre>.npz`."""
    asegurar_carpetas()
    ruta = DATA_PROCESSED / f"{nombre}.npz"
    np.savez_compressed(ruta, **arreglos)
    return ruta


def cargar_npz(nombre: str) -> dict[str, np.ndarray]:
    """Lee `data/processed/<nombre>.npz` y devuelve sus arreglos."""
    ruta = DATA_PROCESSED / f"{nombre}.npz"
    with np.load(ruta, allow_pickle=True) as datos:
        return {clave: datos[clave] for clave in datos.files}


def guardar_json(nombre: str, contenido: dict[str, Any]) -> Path:
    """Guarda metadatos en `data/processed/<nombre>.json`."""
    asegurar_carpetas()
    ruta = DATA_PROCESSED / f"{nombre}.json"
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def cargar_json(nombre: str) -> dict[str, Any]:
    """Lee `data/processed/<nombre>.json`."""
    return json.loads((DATA_PROCESSED / f"{nombre}.json").read_text(encoding="utf-8"))
