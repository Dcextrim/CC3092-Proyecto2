"""Descarga de los datos crudos del proyecto a `data/raw`.

Fuente primaria: Kaggle, descargando archivo por archivo para no arrastrar las
seis variantes del dataset de IBM. Requiere credenciales.

Fuente de respaldo: espejo público de Hugging Face, disponible solo para
`HI-Small_Trans.csv`. Se usa únicamente cuando no hay credenciales de Kaggle.

Uso:
    python src/data_pipeline/sourcing/downloaders/descarga.py
    python src/data_pipeline/sourcing/downloaders/descarga.py --solo-ibm
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[4] / "src"))

from utils.rutas import DATA_RAW, asegurar_carpetas  # noqa: E402

IBM_AML_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
IBM_AML_ARCHIVO = "HI-Small_Trans.csv"
IBM_AML_ESPEJO = (
    "https://huggingface.co/datasets/"
    "eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/"
    "resolve/main/HI-Small_Trans.csv.zip"
)

PAYSIM_HANDLE = "ealaxi/paysim1"
PAYSIM_ARCHIVO = "PS_20174392719_1491204439457_log.csv"

AVISO_RESPALDO = (
    "Se usó el espejo de Hugging Face porque no se encontraron credenciales de Kaggle.\n"
    "  La fuente citable del dataset sigue siendo Kaggle. Para reproducir el proyecto\n"
    "  completo, incluida la verificación sobre PaySim, configure las llaves según el\n"
    "  paso de credenciales del README."
)


def hay_credenciales_kaggle() -> bool:
    """Indica si hay credenciales de Kaggle utilizables en este entorno.

    Kaggle admite dos formas, el par usuario y llave del archivo `kaggle.json`,
    y el token mas reciente que `kagglehub` guarda por separado. Cualquiera de
    las dos sirve, asi que se aceptan ambas.
    """
    try:
        from kagglehub.config import get_kaggle_credentials
    except ImportError:
        return False
    credenciales = get_kaggle_credentials()
    if credenciales is None:
        return False
    return bool(
        (credenciales.username and credenciales.key)
        or getattr(credenciales, "api_key", None)
    )


def _descargar_url(url: str, destino: Path, descripcion: str) -> Path:
    """Descarga `url` a `destino` mostrando el avance."""
    asegurar_carpetas()
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    with requests.get(url, stream=True, timeout=120) as respuesta:
        respuesta.raise_for_status()
        total = int(respuesta.headers.get("content-length", 0))
        with parcial.open("wb") as salida, tqdm(
            total=total, unit="B", unit_scale=True, desc=descripcion
        ) as barra:
            for bloque in respuesta.iter_content(chunk_size=1 << 20):
                salida.write(bloque)
                barra.update(len(bloque))
    parcial.replace(destino)
    return destino


def _descargar_de_kaggle(handle: str, archivo: str, destino: Path) -> Path:
    """Descarga un archivo suelto de un dataset de Kaggle hacia `destino`."""
    import kagglehub

    asegurar_carpetas()
    obtenido = Path(kagglehub.dataset_download(handle, path=archivo))
    if obtenido.is_dir():
        obtenido = next(obtenido.rglob(archivo))
    if obtenido.resolve() != destino.resolve():
        shutil.copy2(obtenido, destino)
    return destino


def descargar_ibm_aml(forzar: bool = False) -> tuple[Path, str]:
    """Deja `HI-Small_Trans.csv` en `data/raw` y devuelve su ruta y su origen."""
    destino = DATA_RAW / IBM_AML_ARCHIVO
    if destino.exists() and not forzar:
        return destino, "cache"

    if hay_credenciales_kaggle():
        try:
            _descargar_de_kaggle(IBM_AML_HANDLE, IBM_AML_ARCHIVO, destino)
            return destino, "kaggle"
        except Exception as error:
            print(f"  La descarga desde Kaggle fallo ({type(error).__name__}: {error}).")
            print("  Se intenta con el espejo publico.")

    comprimido = DATA_RAW / "HI-Small_Trans.csv.zip"
    _descargar_url(IBM_AML_ESPEJO, comprimido, "IBM AML (espejo)")
    with zipfile.ZipFile(comprimido) as zf:
        interno = next(n for n in zf.namelist() if n.endswith(IBM_AML_ARCHIVO))
        with zf.open(interno) as origen, destino.open("wb") as salida:
            shutil.copyfileobj(origen, salida)
    comprimido.unlink()
    return destino, "espejo"


def descargar_paysim(forzar: bool = False) -> Path | None:
    """Deja el log de PaySim en `data/raw`, o None si no hay credenciales."""
    destino = DATA_RAW / PAYSIM_ARCHIVO
    if destino.exists() and not forzar:
        return destino
    if not hay_credenciales_kaggle():
        return None
    try:
        return _descargar_de_kaggle(PAYSIM_HANDLE, PAYSIM_ARCHIVO, destino)
    except Exception as error:
        print(f"  La descarga de PaySim fallo ({type(error).__name__}: {error}).")
        return None


def main() -> int:
    """Descarga los dos datasets e informa qué ruta se usó para cada uno."""
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--solo-ibm", action="store_true", help="omitir PaySim")
    analizador.add_argument("--forzar", action="store_true", help="volver a descargar")
    argumentos = analizador.parse_args()

    ruta_ibm, origen = descargar_ibm_aml(forzar=argumentos.forzar)
    print(f"IBM AML  : {ruta_ibm}  [{origen}]  {ruta_ibm.stat().st_size / 1e6:.1f} MB")
    if origen == "espejo":
        print(f"  Aviso: {AVISO_RESPALDO}")

    if not argumentos.solo_ibm:
        ruta_paysim = descargar_paysim(forzar=argumentos.forzar)
        if ruta_paysim is None:
            print("PaySim   : omitido, no hay credenciales de Kaggle")
        else:
            print(f"PaySim   : {ruta_paysim}  {ruta_paysim.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
