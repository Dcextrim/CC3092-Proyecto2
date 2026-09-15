"""Empaquetado de los artefactos que produce un cuaderno.

Cuando los cuadernos corren en Google Colab, todo lo que escriben queda en una
máquina virtual que se destruye al cerrar la sesión. Este módulo junta los
archivos que hay que conservar en un solo zip para descargarlo.

Uso:
    from utils.artefactos import empaquetar

    archivo = empaquetar("notebook_02")
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from utils.rutas import RAIZ

CARPETAS = (
    "results/models",
    "results/plots",
    "results/tables",
    "src/inference/deployment/datos",
)


def empaquetar(nombre: str, carpetas: tuple[str, ...] = CARPETAS) -> Path:
    """Comprime las carpetas indicadas en `<nombre>.zip` en la raíz del proyecto.

    Las rutas dentro del zip son relativas a la raíz, de modo que descomprimirlo
    sobre una copia local del repositorio deja cada archivo en su lugar.
    """
    destino = RAIZ / f"{nombre}.zip"
    incluidos = 0
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as zf:
        for carpeta in carpetas:
            base = RAIZ / carpeta
            if not base.exists():
                continue
            for archivo in sorted(base.rglob("*")):
                if archivo.is_file():
                    zf.write(archivo, archivo.relative_to(RAIZ))
                    incluidos += 1
    print(f"{destino.name}: {incluidos} archivos, {destino.stat().st_size / 1e6:.2f} MB")
    return destino
