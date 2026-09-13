"""Lectura de `config/model_config.yml`.

Expone la configuración como un diccionario anidado y una ayuda de acceso por
ruta con puntos, para que los módulos no repitan literales dispersos.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from utils.rutas import CONFIG


@lru_cache(maxsize=1)
def cargar_config() -> dict[str, Any]:
    """Lee y memoriza `config/model_config.yml`."""
    return yaml.safe_load((CONFIG / "model_config.yml").read_text(encoding="utf-8"))


def obtener(ruta: str, defecto: Any = None) -> Any:
    """Devuelve un valor de la configuración, por ejemplo `secuencias.longitud_maxima`."""
    nodo: Any = cargar_config()
    for clave in ruta.split("."):
        if not isinstance(nodo, dict) or clave not in nodo:
            return defecto
        nodo = nodo[clave]
    return nodo
