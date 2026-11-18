"""
Descarga los pesos de un modelo desde su URL de publicación (GitHub Release)
y verifica su SHA-256 contra el manifiesto versionado en git.

Uso (desde la raíz del repositorio):
    python scripts/descargar_modelo.py                                   # todos los modelos de models/*.manifest.json
    python scripts/descargar_modelo.py --manifiesto models/m2_yolo.manifest.json

Si el hash no coincide, el archivo descargado se borra y el script termina
con error: nunca se deja en models/ un archivo sin verificar.
Solo usa la biblioteca estándar.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_MODELOS = RAIZ / "models"


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(manifiesto_ruta: Path, url: str | None = None, timeout: float = 120) -> Path:
    manifiesto = json.loads(manifiesto_ruta.read_text(encoding="utf-8"))
    destino = manifiesto_ruta.parent / manifiesto["archivo"]
    esperado = manifiesto["sha256"]

    if destino.is_file() and _sha256(destino) == esperado:
        print(f"Ya existe y el hash coincide: {destino}")
        return destino

    url = url or manifiesto.get("url_descarga")
    if not url:
        raise SystemExit(f"El manifiesto {manifiesto_ruta} no tiene url_descarga.")

    temporal = destino.with_suffix(destino.suffix + ".parcial")
    print(f"Descargando {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(temporal, "wb") as f:
            while bloque := resp.read(1 << 20):
                f.write(bloque)
    except Exception as e:
        temporal.unlink(missing_ok=True)
        raise SystemExit(
            f"No se pudo descargar {url}: {e}\n"
            "¿Existe la Release con ese archivo? Ver docs/decisiones/ADR-004-politica-pesos-y-datos.md"
        ) from e

    obtenido = _sha256(temporal)
    if obtenido != esperado:
        temporal.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 incorrecto: esperado {esperado}, obtenido {obtenido}. Archivo descartado.")
    temporal.replace(destino)
    print(f"OK (SHA-256 verificado): {destino}")
    return destino


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifiesto", type=Path, help="un solo modelo (por defecto: todos los de models/)")
    p.add_argument("--url", help="sobrescribe la url_descarga del manifiesto (solo con --manifiesto)")
    p.add_argument("--carpeta", type=Path, default=CARPETA_MODELOS, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.url and not args.manifiesto:
        p.error("--url exige --manifiesto")
    manifiestos = [args.manifiesto] if args.manifiesto else sorted(args.carpeta.glob("*.manifest.json"))
    if not manifiestos:
        raise SystemExit(f"No hay manifiestos en {args.carpeta}")
    for m in manifiestos:
        descargar(m, args.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
