"""
Prepara la publicación del modelo M2 (YOLO) para la app: copia los pesos a models/ y escribe
models/m2_yolo.manifest.json con la url de la GitHub Release. No sube nada: la Release la crea el equipo.

Uso (en Colab o en el PC, desde la raíz del repositorio):
    python scripts/publicar_modelo_m2.py --corrida <carpeta con m2_yolo.pt y m2_yolo.manifest.json>

Comprobaciones (fallan con un mensaje claro; --forzar las salta, y queda anotado en el manifiesto):
  - el SHA-256 de los pesos coincide con el manifiesto de la corrida;
  - la corrida es la oficial de configs/informe.yaml (la que se evaluó en test);
  - la evaluación final en test ya se hizo (validado_con_datos_reales = true).

Después:
  1. GitHub → Releases → "Draft a new release" → tag <tag> → adjuntar models/m2_yolo.pt → Publish.
  2. git add models/m2_yolo.manifest.json && git commit (los pesos NO van a git: .gitignore los excluye).
  3. En otro computador: python scripts/descargar_modelo.py  (descarga y verifica el hash).
Solo usa la biblioteca estándar y PyYAML.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REPO_POR_DEFECTO = "Darwin-md11/rendimientocosecha"
TAG_POR_DEFECTO = "modelo-m2-v1"
ARCHIVO = "m2_yolo.pt"
LICENCIA = "TODO(fuente): pendiente de verificar; ver docs/legal.md (Ultralytics y Croppie CC BY-SA 4.0)"


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def corrida_oficial(config: Path) -> str:
    import yaml

    return yaml.safe_load(config.read_text(encoding="utf-8"))["corridas"]["m2"]


def preparar(
    corrida: Path,
    destino: Path,
    repo: str = REPO_POR_DEFECTO,
    tag: str = TAG_POR_DEFECTO,
    config: Path = RAIZ / "configs" / "informe.yaml",
    forzar: bool = False,
) -> Path:
    pesos, man_ruta = corrida / ARCHIVO, corrida / "m2_yolo.manifest.json"
    for r in (pesos, man_ruta):
        if not r.is_file():
            raise SystemExit(f"No existe {r}. Descargue de Drive la carpeta de la corrida (o al menos esos dos archivos).")
    man = json.loads(man_ruta.read_text(encoding="utf-8"))
    sha = _sha256(pesos)
    if sha != man.get("sha256"):  # esta comprobación no se puede saltar
        raise SystemExit(f"El SHA-256 de {pesos} ({sha[:12]}…) no coincide con su manifiesto ({str(man.get('sha256'))[:12]}…).")
    problemas = []
    oficial = corrida_oficial(config)
    if man.get("corrida_id") != oficial:
        problemas.append(f"la corrida {man.get('corrida_id')!r} no es la oficial de {config.name} ({oficial!r})")
    if not man.get("validado_con_datos_reales"):
        problemas.append("falta la evaluación final en test (python -m ml.m2_yolo.evaluar --split test --final)")
    if problemas and not forzar:
        raise SystemExit("No se prepara la publicación: " + "; ".join(problemas) + ". Use --forzar solo si el equipo lo decide.")

    destino.mkdir(parents=True, exist_ok=True)
    if pesos.resolve() != (destino / ARCHIVO).resolve():
        shutil.copy2(pesos, destino / ARCHIVO)
    publicado = {
        **man,
        "url_descarga": f"https://github.com/{repo}/releases/download/{tag}/{ARCHIVO}",
        "release_tag": tag,
        "licencia": man.get("licencia") or LICENCIA,
    }
    if problemas:
        publicado["advertencias_publicacion"] = problemas
    salida = destino / "m2_yolo.manifest.json"
    salida.write_text(json.dumps(publicado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return salida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrida", type=Path, required=True)
    p.add_argument("--destino", type=Path, default=RAIZ / "models")
    p.add_argument("--repo", default=REPO_POR_DEFECTO)
    p.add_argument("--tag", default=TAG_POR_DEFECTO)
    p.add_argument("--config", type=Path, default=RAIZ / "configs" / "informe.yaml")
    p.add_argument("--forzar", action="store_true")
    a = p.parse_args(argv)
    salida = preparar(a.corrida, a.destino, a.repo, a.tag, a.config, a.forzar)
    man = json.loads(salida.read_text(encoding="utf-8"))
    print(f"Listo: {a.destino / ARCHIVO} (SHA-256 {man['sha256'][:12]}…) y {salida}")
    print(f"1. Cree la Release '{a.tag}' en https://github.com/{a.repo}/releases/new y adjunte {ARCHIVO}.")
    print(f'2. git add {salida.relative_to(RAIZ) if salida.is_relative_to(RAIZ) else salida} && git commit -m "feat(modelo): publicar M2 {a.tag}"')
    print("3. En otro computador: python scripts/descargar_modelo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
