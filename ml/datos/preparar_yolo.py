"""
Prepara un dataset de detección en formato YOLO para SIKAFE (CRISP-DM 2–3).

  1. Descubre imágenes y etiquetas YOLO (cualquier estructura de carpetas).
  2. Mapea las clases del dataset a las 4 de SIKAFE (configs/clases.yaml).
     Las clases sin mapeo se excluyen; por defecto su región se ENMASCARA en la
     imagen (si solo se borrara la caja, el modelo aprendería esa cereza como fondo).
  3. Agrupa los recortes que vienen de la misma foto original y reparte 70/15/15
     POR GRUPO con semilla fija (evita fuga de datos entre train y test).
  4. Prueba de fuga: ningún grupo ni archivo idéntico (SHA-256) en dos particiones.
  5. Escribe data.yaml (Ultralytics), manifiesto con hashes y estadísticas (EDA).

Uso:
    python -m ml.datos.preparar_yolo --origen /content/croppie_raw --inspeccionar
    python -m ml.datos.preparar_yolo --origen /content/croppie_raw --destino /content/sikafe_yolo \
        --dataset croppie --semilla 0
"""

import argparse
import collections
import hashlib
import json
import random
import re
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image, ImageDraw, ImageOps

RAIZ = Path(__file__).resolve().parents[2]
CONFIG_CLASES = RAIZ / "configs" / "clases.yaml"
EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROPORCIONES = {"train": 0.70, "val": 0.15, "test": 0.15}

# Candidatos para quitar el sufijo de recorte y recuperar la foto original (se prueban en orden).
# Antes de aplicarlos siempre se quita el sufijo de Roboflow ".rf.<hash>" y la extensión.
_TIPO = r"(?:tile|crop|patch|part|piece)"
CANDIDATOS_GRUPO = {
    "fila_columna": rf"[_-]?{_TIPO}?[_-]?\d+[_-]\d+$",  # foto_2_3, foto-tile_2_3
    "indice": rf"[_-]?{_TIPO}?[_-]?\d+$",  # foto_7, foto-tile7  (OJO: también corta IMG_1234)
}
PATRON_AUTO = "auto"


class ErrorPreparacion(Exception):
    pass


@dataclass
class Muestra:
    imagen: Path
    etiqueta: Optional[Path]
    split_origen: str
    grupo: str = ""
    cajas: List[Tuple[int, float, float, float, float]] = field(default_factory=list)  # clase SIKAFE, cx, cy, w, h
    excluidas: List[Tuple[str, float, float, float, float]] = field(default_factory=list)
    sha256: str = ""
    nombre_destino: str = ""


# ------------------------------------------------------------------ configuración
def cargar_mapeo(dataset: str, ruta_config: Path = CONFIG_CLASES):
    cfg = yaml.safe_load(ruta_config.read_text(encoding="utf-8"))
    clases = list(cfg["clases_sikafe"])
    if dataset not in cfg["datasets"]:
        raise ErrorPreparacion(f"Dataset {dataset!r} no está en {ruta_config}. Disponibles: {list(cfg['datasets'])}")
    d = cfg["datasets"][dataset]
    origen = {int(k): v for k, v in d["clases_origen"].items()}
    mapeo: Dict[int, Optional[int]] = {}
    for idx, nombre in origen.items():
        if nombre not in d["mapeo"]:
            raise ErrorPreparacion(f"La clase {nombre!r} no tiene regla en el mapeo de {dataset} (use null para excluirla).")
        destino = d["mapeo"][nombre]
        if destino is not None and destino not in clases:
            raise ErrorPreparacion(f"{nombre!r} se mapea a {destino!r}, que no es una clase SIKAFE {clases}.")
        mapeo[idx] = None if destino is None else clases.index(destino)
    return clases, origen, mapeo, d


# ------------------------------------------------------------------ descubrimiento
def _split_de(ruta: Path) -> str:
    partes = [p.lower() for p in ruta.parts]
    for p in reversed(partes):
        if p in ("train", "training"):
            return "train"
        if p in ("val", "valid", "validation"):
            return "val"
        if p == "test":
            return "test"
    return "sin_split"


def _etiqueta_de(imagen: Path) -> Optional[Path]:
    candidatos = [imagen.with_suffix(".txt")]
    partes = list(imagen.parts)
    for i in range(len(partes) - 1, -1, -1):
        if partes[i].lower() == "images":
            alt = Path(*partes[:i], "labels", *partes[i + 1 :]).with_suffix(".txt")
            candidatos.append(alt)
            break
    return next((c for c in candidatos if c.is_file()), None)


def descubrir(origen: Path) -> List[Muestra]:
    imagenes = sorted(p for p in origen.rglob("*") if p.suffix.lower() in EXT_IMAGEN and p.is_file())
    if not imagenes:
        raise ErrorPreparacion(f"No hay imágenes en {origen}")
    return [Muestra(imagen=p, etiqueta=_etiqueta_de(p), split_origen=_split_de(p.relative_to(origen))) for p in imagenes]


def clave_grupo(nombre_archivo: str, patron: str) -> str:
    """Identificador de la foto original de la que sale un recorte (se borra `patron` del nombre)."""
    base = re.sub(r"\.rf\.[0-9a-f]+$", "", Path(nombre_archivo).stem)
    return re.sub(patron, "", base) or base


def _plausible(claves: List[str]) -> Tuple[bool, str]:
    n = len(claves)
    conteo = collections.Counter(claves)
    mayor = conteo.most_common(1)[0][1]
    g = len(conteo)
    if g == n:
        return False, "no une ningún recorte"
    if g < 20:
        return False, f"solo {g} grupos"
    if mayor > max(64, 0.05 * n):
        return False, f"un grupo tiene {mayor} imágenes (demasiado: el patrón borra de más)"
    return True, f"{g} grupos, el mayor con {mayor} imágenes"


def elegir_patron(nombres: List[str], patron: str = PATRON_AUTO) -> Tuple[str, str, Dict[str, str]]:
    """Devuelve (nombre_patron, regex, diagnóstico por candidato). Lanza ErrorPreparacion si nada es plausible."""
    candidatos = CANDIDATOS_GRUPO if patron == PATRON_AUTO else {"manual": patron}
    diagnostico = {}
    for nombre, regex in candidatos.items():
        ok, motivo = _plausible([clave_grupo(x, regex) for x in nombres])
        diagnostico[nombre] = motivo
        if ok:
            return nombre, regex, diagnostico
    raise ErrorPreparacion(
        f"Ningún patrón de agrupamiento es plausible: {diagnostico}. Revise los nombres con --inspeccionar y pase "
        "--regex-grupo, o use --usar-split-original (y declare la limitación)."
    )


# ------------------------------------------------------------------ etiquetas
def leer_etiquetas(m: Muestra, origen_clases: Dict[int, str], mapeo: Dict[int, Optional[int]]) -> None:
    if m.etiqueta is None:
        return
    for n, linea in enumerate(m.etiqueta.read_text(encoding="utf-8").splitlines(), start=1):
        if not linea.strip():
            continue
        partes = linea.split()
        if len(partes) != 5:
            raise ErrorPreparacion(f"{m.etiqueta}:{n}: se esperaban 5 valores (formato YOLO de cajas), hay {len(partes)}")
        c = int(float(partes[0]))
        cx, cy, w, h = (float(v) for v in partes[1:])
        if c not in mapeo:
            raise ErrorPreparacion(f"{m.etiqueta}:{n}: clase {c} desconocida para este dataset")
        if w <= 0 or h <= 0:
            continue
        if mapeo[c] is None:
            m.excluidas.append((origen_clases[c], cx, cy, w, h))
        else:
            m.cajas.append((mapeo[c], cx, cy, w, h))


# ------------------------------------------------------------------ partición
def particionar(muestras: List[Muestra], semilla: int) -> Dict[str, str]:
    """Asigna cada GRUPO a train/val/test (~70/15/15 en número de imágenes)."""
    por_grupo = collections.Counter(m.grupo for m in muestras)
    grupos = sorted(por_grupo)
    random.Random(semilla).shuffle(grupos)
    total = sum(por_grupo.values())
    asignacion, acumulado = {}, 0
    limite_train = PROPORCIONES["train"] * total
    limite_val = (PROPORCIONES["train"] + PROPORCIONES["val"]) * total
    for g in grupos:
        centro = acumulado + por_grupo[g] / 2
        asignacion[g] = "train" if centro <= limite_train else ("val" if centro <= limite_val else "test")
        acumulado += por_grupo[g]
    return asignacion


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def verificar_fuga(muestras: List[Muestra], split_de: Dict[str, str]) -> Dict[str, int]:
    grupos_por_split = collections.defaultdict(set)
    for m in muestras:
        grupos_por_split[split_de[m.nombre_destino]].add(m.grupo)
    splits = list(grupos_por_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            comun = grupos_por_split[a] & grupos_por_split[b]
            if comun:
                raise ErrorPreparacion(f"FUGA: {len(comun)} grupos en {a} y {b}, p. ej. {sorted(comun)[:3]}")
    split_por_hash: Dict[str, str] = {}
    for m in muestras:
        s = split_de[m.nombre_destino]
        previo = split_por_hash.setdefault(m.sha256, s)
        if previo != s:
            raise ErrorPreparacion(f"FUGA: la imagen {m.imagen.name} está repetida (mismo SHA-256) en {previo} y {s}")
    repetidos = collections.Counter(m.sha256 for m in muestras)
    duplicados_mismo_split = sum(n - 1 for n in repetidos.values() if n > 1)
    return {"grupos_por_split": {s: len(g) for s, g in grupos_por_split.items()}, "duplicados_mismo_split": duplicados_mismo_split}


# ------------------------------------------------------------------ escritura
def _escribir_imagen(m: Muestra, destino: Path, enmascarar: bool) -> Tuple[int, int]:
    with Image.open(m.imagen) as im:
        tam = im.size
        if enmascarar and m.excluidas:
            im = ImageOps.exif_transpose(im).convert("RGB")
            tam = im.size
            dibujo = ImageDraw.Draw(im)
            for _, cx, cy, w, h in m.excluidas:
                x0, y0 = (cx - w / 2) * tam[0], (cy - h / 2) * tam[1]
                x1, y1 = (cx + w / 2) * tam[0], (cy + h / 2) * tam[1]
                dibujo.rectangle([x0, y0, x1, y1], fill=(114, 114, 114))  # gris de relleno de Ultralytics
            im.save(destino, quality=95)
            return tam
    shutil.copy2(m.imagen, destino)
    return tam


def preparar(
    origen: Path,
    destino: Path,
    dataset: str,
    semilla: int = 0,
    patron_grupo: str = PATRON_AUTO,
    usar_split_original: bool = False,
    enmascarar_excluidas: bool = True,
    ruta_config: Path = CONFIG_CLASES,
) -> dict:
    clases, origen_clases, mapeo, meta = cargar_mapeo(dataset, ruta_config)
    muestras = descubrir(origen)
    sin_etiqueta = sum(1 for m in muestras if m.etiqueta is None)
    if usar_split_original:
        nombre_patron, regex, diagnostico = "ninguno", "", {}
    else:
        nombre_patron, regex, diagnostico = elegir_patron([m.imagen.name for m in muestras], patron_grupo)
    usados = set()
    for m in muestras:
        leer_etiquetas(m, origen_clases, mapeo)
        m.grupo = clave_grupo(m.imagen.name, regex) if regex else m.imagen.stem
        m.sha256 = _sha256(m.imagen)
        nombre = m.imagen.name if m.imagen.name not in usados else f"{m.split_origen}_{m.imagen.name}"
        usados.add(nombre)
        m.nombre_destino = nombre

    n_grupos = len({m.grupo for m in muestras})
    if usar_split_original:
        if any(m.split_origen == "sin_split" for m in muestras):
            raise ErrorPreparacion("Hay imágenes sin carpeta train/val/test: no se puede usar la partición original.")
        split_de = {m.nombre_destino: m.split_origen for m in muestras}
        estrategia = "partición original del dataset (agrupamiento no verificado)"
    else:
        asignacion = particionar(muestras, semilla)
        split_de = {m.nombre_destino: asignacion[m.grupo] for m in muestras}
        estrategia = f"70/15/15 por grupo (foto original; patrón {nombre_patron}: {diagnostico[nombre_patron]}), semilla {semilla}"
    fuga = verificar_fuga(muestras, split_de)

    if destino.exists():
        shutil.rmtree(destino)
    tamanos = collections.Counter()
    for m in muestras:
        s = split_de[m.nombre_destino]
        (destino / "images" / s).mkdir(parents=True, exist_ok=True)
        (destino / "labels" / s).mkdir(parents=True, exist_ok=True)
        tam = _escribir_imagen(m, destino / "images" / s / m.nombre_destino, enmascarar_excluidas)
        tamanos[f"{tam[0]}x{tam[1]}"] += 1
        lineas = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in m.cajas]
        (destino / "labels" / s / Path(m.nombre_destino).with_suffix(".txt").name).write_text(
            "\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8"
        )

    data_yaml = {
        "path": str(destino.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(clases)},
    }
    (destino / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    registros = sorted(
        (
            {
                "archivo": m.nombre_destino,
                "origen": str(m.imagen.relative_to(origen)),
                "split": split_de[m.nombre_destino],
                "grupo": m.grupo,
                "sha256": m.sha256,
                "cajas": len(m.cajas),
                "excluidas": len(m.excluidas),
            }
            for m in muestras
        ),
        key=lambda r: r["archivo"],
    )
    version = hashlib.sha256(json.dumps([(r["archivo"], r["sha256"], r["split"]) for r in registros]).encode()).hexdigest()
    stats = estadisticas(muestras, split_de, clases, tamanos)
    resumen = {
        "dataset": dataset,
        "url": meta.get("url"),
        "licencia": meta.get("licencia"),
        "atribucion": meta.get("atribucion"),
        "version_dataset_sha256": version,
        "estrategia_particion": estrategia,
        "semilla": semilla,
        "patron_grupo": {"nombre": nombre_patron, "regex": regex, "diagnostico": diagnostico},
        "grupos": n_grupos,
        "imagenes": len(muestras),
        "imagenes_sin_etiqueta": sin_etiqueta,
        "excluidas_enmascaradas": enmascarar_excluidas,
        "mapeo": {origen_clases[k]: (None if v is None else clases[v]) for k, v in mapeo.items()},
        "prueba_fuga": "OK",
        **fuga,
        "estadisticas": stats,
    }
    (destino / "manifiesto.json").write_text(json.dumps({"resumen": resumen, "archivos": registros}, indent=1, ensure_ascii=False), encoding="utf-8")
    (destino / "estadisticas.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    (destino / "eda.md").write_text(eda_markdown(resumen), encoding="utf-8")
    _grafico_clases(stats, destino / "eda_clases.png")
    return resumen


# ------------------------------------------------------------------ EDA
def estadisticas(muestras, split_de, clases, tamanos) -> dict:
    por_split = {}
    for s in ("train", "val", "test"):
        ms = [m for m in muestras if split_de[m.nombre_destino] == s]
        cajas = [c for m in ms for c in m.cajas]
        por_clase = collections.Counter(clases[c[0]] for c in cajas)
        excl = collections.Counter(e[0] for m in ms for e in m.excluidas)
        n_por_img = [len(m.cajas) for m in ms]
        lados = [max(c[3], c[4]) for c in cajas]
        por_split[s] = {
            "imagenes": len(ms),
            "imagenes_sin_cajas": sum(1 for n in n_por_img if n == 0),
            "cajas": len(cajas),
            "cajas_por_clase": {c: por_clase.get(c, 0) for c in clases},
            "excluidas_por_clase": dict(excl),
            "cajas_por_imagen": {
                "media": round(statistics.mean(n_por_img), 2) if n_por_img else 0,
                "mediana": statistics.median(n_por_img) if n_por_img else 0,
                "max": max(n_por_img) if n_por_img else 0,
            },
            "lado_mayor_caja_relativo": {
                "mediana": round(statistics.median(lados), 4) if lados else None,
                "p10": round(sorted(lados)[len(lados) // 10], 4) if lados else None,
            },
        }
    return {"por_split": por_split, "tamanos_imagen": dict(tamanos.most_common(10))}


def eda_markdown(r: dict) -> str:
    st = r["estadisticas"]["por_split"]
    clases = list(next(iter(st.values()))["cajas_por_clase"])
    filas = [
        "| Partición | Imágenes | Sin cajas | Cajas | " + " | ".join(clases) + " | Cajas/imagen (media) | Lado mayor caja (mediana, rel.) |",
        "|---" * (6 + len(clases)) + "|",
    ]
    for s, d in st.items():
        filas.append(
            f"| {s} | {d['imagenes']} | {d['imagenes_sin_cajas']} | {d['cajas']} | "
            + " | ".join(str(d["cajas_por_clase"][c]) for c in clases)
            + f" | {d['cajas_por_imagen']['media']} | {d['lado_mayor_caja_relativo']['mediana']} |"
        )
    excl = collections.Counter()
    for d in st.values():
        excl.update(d["excluidas_por_clase"])
    return "\n".join(
        [
            f"# Análisis exploratorio — {r['dataset']}",
            "",
            f"- Fuente: {r['url']} · Licencia: {r['licencia']} · {r['atribucion']}",
            f"- Versión del dataset preparado (SHA-256 del manifiesto): `{r['version_dataset_sha256']}`",
            f"- Partición: {r['estrategia_particion']} · grupos: {r['grupos']} · imágenes: {r['imagenes']} "
            f"(sin archivo de etiquetas: {r['imagenes_sin_etiqueta']})",
            f"- Prueba de fuga: {r['prueba_fuga']} (grupos por partición: {r['grupos_por_split']})",
            f"- Mapeo de clases: {r['mapeo']}",
            f"- Cajas excluidas ({'enmascaradas en la imagen' if r['excluidas_enmascaradas'] else 'solo eliminadas'}): {dict(excl)}",
            f"- Tamaños de imagen más frecuentes: {r['estadisticas']['tamanos_imagen']}",
            "",
            *filas,
            "",
            "Generado por `python -m ml.datos.preparar_yolo` (no editar a mano).",
            "",
        ]
    )


def _grafico_clases(stats: dict, ruta: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("AVISO: matplotlib no está instalado; no se generan las figuras (pip install matplotlib).")
        return
    st = stats["por_split"]
    clases = list(st["train"]["cajas_por_clase"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ancho = 0.27
    for i, s in enumerate(("train", "val", "test")):
        ax.bar([j + (i - 1) * ancho for j in range(len(clases))], [st[s]["cajas_por_clase"][c] for c in clases], ancho, label=s)
    ax.set_xticks(range(len(clases)), clases)
    ax.set_ylabel("Cajas (cerezas)")
    ax.set_title("Distribución de clases por partición")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ inspección
def inspeccionar(origen: Path, patron_grupo: str = PATRON_AUTO, n: int = 15) -> dict:
    muestras = descubrir(origen)
    por_split = collections.Counter(m.split_origen for m in muestras)
    try:
        nombre_patron, regex, diagnostico = elegir_patron([m.imagen.name for m in muestras], patron_grupo)
    except ErrorPreparacion as e:
        nombre_patron, regex, diagnostico = "NINGUNO", r"$^", {"error": str(e)}
    grupos = collections.defaultdict(list)
    for m in muestras:
        grupos[clave_grupo(m.imagen.name, regex)].append(m)
    mezclados = sum(1 for ms in grupos.values() if len({m.split_origen for m in ms}) > 1)
    clases = collections.Counter()
    for m in muestras:
        if m.etiqueta:
            clases.update(linea.split()[0] for linea in m.etiqueta.read_text(encoding="utf-8").splitlines() if linea.strip())
    info = {
        "imagenes": len(muestras),
        "por_split_origen": dict(por_split),
        "sin_etiqueta": sum(1 for m in muestras if m.etiqueta is None),
        "patron_elegido": nombre_patron,
        "diagnostico_patrones": diagnostico,
        "grupos_con_patron": len(grupos),
        "grupos_en_varios_splits_originales": mezclados,
        "cajas_por_clase_origen": dict(sorted(clases.items())),
        "ejemplos": [
            f"{m.imagen.relative_to(origen)} -> grupo {clave_grupo(m.imagen.name, regex)!r}" for m in muestras[:: max(1, len(muestras) // n)][:n]
        ],
    }
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origen", type=Path, required=True)
    p.add_argument("--destino", type=Path)
    p.add_argument("--dataset", default="croppie")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--regex-grupo", default=PATRON_AUTO, help="regex que se BORRA del nombre para obtener la foto original (auto)")
    p.add_argument("--usar-split-original", action="store_true", help="respetar train/val del dataset (sin reagrupar)")
    p.add_argument("--no-enmascarar", action="store_true", help="solo borrar las cajas excluidas, sin taparlas en la imagen")
    p.add_argument("--inspeccionar", action="store_true", help="solo mostrar estructura, nombres y agrupamiento")
    a = p.parse_args(argv)
    try:
        if a.inspeccionar:
            print(json.dumps(inspeccionar(a.origen, a.regex_grupo), indent=2, ensure_ascii=False))
            return 0
        if not a.destino:
            p.error("--destino es obligatorio (salvo con --inspeccionar)")
        preparar(a.origen, a.destino, a.dataset, a.semilla, a.regex_grupo, a.usar_split_original, not a.no_enmascarar)
    except ErrorPreparacion as e:
        print(f"ERROR: {e}")
        return 2
    print((a.destino / "eda.md").read_text(encoding="utf-8"))
    print(f"data.yaml: {a.destino / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
