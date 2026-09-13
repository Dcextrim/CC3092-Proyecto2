"""Orquestador del pipeline de datos, desde el CSV crudo hasta los tensores.

Encadena descarga, ventanas, variables, tensorización, partición y escala, y
guarda el resultado en `data/processed` para que el segundo notebook no tenga
que repetir el trabajo. Si el caché no existe, lo reconstruye, de modo que
cualquiera de los dos notebooks corre por sí solo.

Uso:
    from data_pipeline.construir import construir_o_cargar

    datos = construir_o_cargar()
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.preprocessing import features, secuencias, splits
from data_pipeline.sourcing.downloaders import descarga
from utils.rutas import DATA_PROCESSED, asegurar_carpetas

NOMBRE_CACHE = "secuencias"


@dataclass
class Datos:
    """Todo lo que los notebooks y el MVP necesitan del pipeline de datos."""

    tensores: secuencias.Tensores
    particion: splits.Particion
    escala: splits.Escala
    catalogos: dict[str, list[str]]
    tasas: dict[str, float]


ARREGLOS = (
    "X", "mascara", "y", "longitud", "identificador",
    "entrenamiento", "validacion", "prueba",
    "escala_columnas", "escala_media", "escala_desviacion",
)


def _carpeta_cache() -> Path:
    return DATA_PROCESSED / NOMBRE_CACHE


def _rutas_cache() -> list[Path]:
    carpeta = _carpeta_cache()
    return [carpeta / f"{nombre}.npy" for nombre in ARREGLOS] + [
        carpeta / "contexto.parquet", carpeta / "meta.json"
    ]


def construir(guardar: bool = True) -> Datos:
    """Ejecuta el pipeline completo desde el CSV crudo."""
    ruta, _ = descarga.descargar_ibm_aml()

    crudo, catalogos = features.cargar_transacciones(ruta)
    tasas = features.tasas_a_usd(crudo, catalogos)

    ventana = features.construir_ventanas(crudo)
    del crudo
    gc.collect()

    tabla = features.construir_variables(ventana, catalogos, tasas)
    del ventana
    gc.collect()

    tensores = secuencias.tensorizar(tabla, liberar=True)
    del tabla
    gc.collect()

    particion = splits.particionar(tensores.y)
    escala = splits.ajustar_escala(tensores, particion.entrenamiento)
    splits.aplicar_escala(tensores, escala)

    datos = Datos(
        tensores=tensores, particion=particion, escala=escala,
        catalogos=catalogos, tasas=tasas,
    )
    if guardar:
        guardar_cache(datos)
    return datos


def guardar_cache(datos: Datos) -> None:
    """Escribe cada tensor como un `.npy` independiente en `data/processed`.

    Se usan archivos sueltos y no un `.npz` porque `np.save` escribe en flujo,
    mientras que el empaquetado retiene una copia completa del arreglo más
    grande en memoria.
    """
    asegurar_carpetas()
    carpeta = _carpeta_cache()
    carpeta.mkdir(parents=True, exist_ok=True)
    tensores = datos.tensores
    contenido = {
        "X": tensores.X,
        "mascara": tensores.mascara,
        "y": tensores.y,
        "longitud": tensores.longitud,
        "identificador": tensores.identificador.astype("U16"),
        "entrenamiento": datos.particion.entrenamiento,
        "validacion": datos.particion.validacion,
        "prueba": datos.particion.prueba,
        "escala_columnas": np.asarray(datos.escala.columnas),
        "escala_media": datos.escala.media,
        "escala_desviacion": datos.escala.desviacion,
    }
    for nombre, arreglo in contenido.items():
        np.save(carpeta / f"{nombre}.npy", arreglo)
    tensores.contexto.to_parquet(carpeta / "contexto.parquet", index=False)
    (carpeta / "meta.json").write_text(
        json.dumps(
            {"catalogos": datos.catalogos, "tasas": datos.tasas, "variables": features.VARIABLES},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def cargar_cache(memoria_compartida: bool = False) -> Datos:
    """Lee los tensores guardados por `guardar_cache`.

    Con `memoria_compartida` activo, `X` se abre como mapa de archivo en lugar
    de copiarse a memoria, útil en entornos con poca RAM.
    """
    carpeta = _carpeta_cache()
    leer = lambda nombre, mapa=None: np.load(carpeta / f"{nombre}.npy", mmap_mode=mapa)

    tensores = secuencias.Tensores(
        X=leer("X", "r" if memoria_compartida else None),
        mascara=leer("mascara"),
        y=leer("y"),
        longitud=leer("longitud"),
        identificador=leer("identificador").astype(object),
        contexto=pd.read_parquet(carpeta / "contexto.parquet"),
    )
    particion = splits.Particion(
        entrenamiento=leer("entrenamiento"),
        validacion=leer("validacion"),
        prueba=leer("prueba"),
    )
    escala = splits.Escala(
        columnas=leer("escala_columnas").tolist(),
        media=leer("escala_media"),
        desviacion=leer("escala_desviacion"),
    )
    guardado = json.loads((carpeta / "meta.json").read_text(encoding="utf-8"))
    return Datos(
        tensores=tensores, particion=particion, escala=escala,
        catalogos=guardado["catalogos"], tasas=guardado["tasas"],
    )


def construir_o_cargar(forzar: bool = False, memoria_compartida: bool = False) -> Datos:
    """Devuelve los datos desde el caché, o los construye si hace falta."""
    if not forzar and all(ruta.exists() for ruta in _rutas_cache()):
        return cargar_cache(memoria_compartida=memoria_compartida)
    return construir()


def verificar(datos: Datos) -> dict[str, object]:
    """Comprueba las invariantes del pipeline y devuelve un resumen."""
    tensores, particion = datos.tensores, datos.particion
    conjuntos = [particion.entrenamiento, particion.validacion, particion.prueba]

    total = sum(len(c) for c in conjuntos)
    if total != len(tensores.y):
        raise AssertionError("La particion no cubre a todos los remitentes")
    if len(np.unique(np.concatenate(conjuntos))) != total:
        raise AssertionError("Hay remitentes repetidos entre conjuntos")
    if not np.array_equal(tensores.mascara.sum(axis=1), tensores.longitud):
        raise AssertionError("La mascara no coincide con la longitud de las secuencias")
    relleno = ~tensores.mascara
    for indice in range(tensores.X.shape[2]):
        columna = tensores.X[:, :, indice]
        if np.abs(columna[relleno]).max(initial=0.0) != 0:
            raise AssertionError("Hay relleno con valores distintos de cero")
        if not np.isfinite(columna).all():
            raise AssertionError("Hay valores no finitos en las secuencias")

    return {
        "remitentes": int(len(tensores.y)),
        "transacciones": int(tensores.longitud.sum()),
        "positivos": int(tensores.y.sum()),
        "prevalencia": float(tensores.y.mean()),
        "variables": len(features.VARIABLES),
        "conjuntos": particion.resumen(tensores.y),
    }
