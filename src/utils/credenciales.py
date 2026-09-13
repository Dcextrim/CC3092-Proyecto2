"""Configuración de las credenciales de Kaggle en cualquier entorno.

Kaggle admite varias formas de autenticar y no todas estan disponibles en todos
los entornos. Esta función prueba las vías en orden y explica qué hacer si
ninguna sirve.

Uso:
    from utils.credenciales import configurar_kaggle

    configurar_kaggle()
"""

from __future__ import annotations

import json
import os
from pathlib import Path

VARIABLES = ("KAGGLE_USERNAME", "KAGGLE_KEY")
ARCHIVO = Path.home() / ".kaggle" / "kaggle.json"

INSTRUCCIONES = """
Sin credenciales de Kaggle el proyecto queda incompleto. La descarga del archivo
de IBM AML cae a un espejo público, pero la verificación sobre PaySim se omite.

Para configurarlas, genere un token en la configuracion de su cuenta de Kaggle y
elija una de estas dos vias.

  1. Secretos del cuaderno. En Colab, con el icono de la llave, agregue
     KAGGLE_USERNAME y KAGGLE_KEY, y habilite el acceso para este cuaderno.

  2. Una celda con sus llaves, si los secretos no estan disponibles.

       from utils.credenciales import guardar_kaggle
       guardar_kaggle("su_usuario", "su_llave_de_32_caracteres")
"""


def _hay_credenciales() -> bool:
    """Indica si `kagglehub` ya encuentra credenciales utilizables."""
    try:
        from kagglehub.config import get_kaggle_credentials
    except ImportError:
        return False
    credenciales = get_kaggle_credentials()
    if credenciales is None:
        return False
    return bool(
        (credenciales.username and credenciales.key) or getattr(credenciales, "api_key", None)
    )


def guardar_kaggle(usuario: str, llave: str) -> Path:
    """Escribe `~/.kaggle/kaggle.json` con las llaves indicadas."""
    ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVO.write_text(json.dumps({"username": usuario, "key": llave}), encoding="utf-8")
    ARCHIVO.chmod(0o600)
    os.environ["KAGGLE_USERNAME"] = usuario
    os.environ["KAGGLE_KEY"] = llave
    print(f"Credenciales guardadas en {ARCHIVO}")
    return ARCHIVO


def configurar_kaggle(silencioso: bool = False) -> bool:
    """Deja las credenciales listas si las encuentra, y devuelve si lo logro.

    Prueba en orden las variables de entorno, el archivo `kaggle.json`, y los
    secretos del cuaderno de Colab.
    """
    if all(os.environ.get(nombre) for nombre in VARIABLES):
        return True

    if ARCHIVO.exists():
        try:
            datos = json.loads(ARCHIVO.read_text(encoding="utf-8"))
            if datos.get("username") and datos.get("key"):
                os.environ["KAGGLE_USERNAME"] = datos["username"]
                os.environ["KAGGLE_KEY"] = datos["key"]
                return True
        except (json.JSONDecodeError, OSError):
            pass

    try:
        from google.colab import userdata

        for nombre in VARIABLES:
            os.environ[nombre] = userdata.get(nombre)
        if all(os.environ.get(nombre) for nombre in VARIABLES):
            return True
    except Exception:  # noqa: BLE001
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)

    if _hay_credenciales():
        return True

    if not silencioso:
        print(INSTRUCCIONES)
    return False
