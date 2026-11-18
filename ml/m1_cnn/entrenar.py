"""
M1 — CNN con transfer learning (MobileNetV2 / EfficientNet-B0 / ResNet-50, preentrenadas en ImageNet)
para clasificar la madurez de los recortes de cerezas: los MISMOS recortes y la misma partición que M0.

Uso (Colab con T4 GPU, desde la raíz del repo):
    python -m ml.m1_cnn.entrenar --config ml/experiments/configs/m1_mobilenetv2_96.yaml \
        --datos /content/sikafe_yolo/data.yaml --salida <corridas> --rapido      # prueba de humo (~2 min)
    python -m ml.m1_cnn.entrenar --config ml/experiments/configs/m1_mobilenetv2_96.yaml \
        --datos /content/sikafe_yolo/data.yaml --salida <corridas>               # corrida completa
    python -m ml.m1_cnn.entrenar --corrida <corridas>/<id> --final               # test, UNA sola vez

Estrategia: (1) se congela la red y se entrena solo la cabeza; (2) se descongela todo y se ajusta con
tasa de aprendizaje menor y decaimiento coseno. Pérdida con pesos inversos a la frecuencia de clase.
Aumento de datos en la GPU: volteos, rotaciones de 90° y brillo/contraste leves (SIN tocar el tono ni la
saturación: el color es justamente lo que distingue los estados). Se guarda la época con mejor F1 macro
en val (por eso val es optimista; el test se toca una sola vez con --final).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ml.datos.lectura_yolo import NOMBRES, version_dataset
from ml.experiments import reporte_clasificacion as R
from ml.experiments.registro import capturar_entorno, fijar_semillas, guardar_json, id_corrida
from ml.m1_cnn.recortes import cargar, construir_cache, pesos_por_clase
from src.evaluation import metricas as M
from src.infrastructure.ml.manifiesto import ruta_manifiesto, sha256_archivo, verificar_archivo

MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESV_IMAGENET = (0.229, 0.224, 0.225)
RAPIDO = {"max_train": 2048, "max_val": 1024, "epocas_cabeza": 1, "epocas": 1}


# ------------------------------------------------------------------ modelo (PyTorch, import diferido)
def construir_modelo(arquitectura: str, n_clases: int, preentrenado: bool):
    import torch.nn as nn
    import torchvision.models as tvm

    if arquitectura == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if preentrenado else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_clases)
        cabeza = m.classifier
    elif arquitectura == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if preentrenado else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_clases)
        cabeza = m.classifier
    elif arquitectura == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if preentrenado else None)
        m.fc = nn.Linear(m.fc.in_features, n_clases)
        cabeza = m.fc
    else:
        raise ValueError(f"arquitectura desconocida: {arquitectura}")
    return m, cabeza


class _Recortes:
    """Dataset de PyTorch sobre la caché .npy: devuelve (uint8 HWC, clase). La conversión y el aumento se hacen en la GPU.
    Guarda la RUTA y abre el memmap en cada proceso del DataLoader (pasar el memmap copiaría los datos con spawn/forkserver)."""

    def __init__(self, ruta_x, y, indices):
        self.ruta_x, self.y, self.indices = str(ruta_x), np.asarray(y), np.asarray(indices)
        self._X = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        import torch

        if self._X is None:
            self._X = np.load(self.ruta_x, mmap_mode="r")
        i = int(self.indices[k])
        return torch.from_numpy(np.array(self._X[i])), int(self.y[i])


def _a_tensor(lote_uint8, dispositivo, aumentar: bool, generador=None):
    """(B,H,W,3) uint8 -> (B,3,H,W) float normalizado; con aumento opcional (solo geometría, brillo y contraste)."""
    import torch

    x = lote_uint8.to(dispositivo, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    if aumentar:
        b = x.shape[0]
        voltear_h = torch.rand(b, device=dispositivo, generator=generador) < 0.5
        x = torch.where(voltear_h[:, None, None, None], x.flip(3), x)
        voltear_v = torch.rand(b, device=dispositivo, generator=generador) < 0.5
        x = torch.where(voltear_v[:, None, None, None], x.flip(2), x)
        k = int(torch.randint(0, 4, (1,), device=dispositivo, generator=generador).item())  # misma rotación para el lote
        x = torch.rot90(x, k, dims=(2, 3))
        brillo = 1 + (torch.rand(b, 1, 1, 1, device=dispositivo, generator=generador) - 0.5) * 0.4  # ±20 %
        media = x.mean(dim=(1, 2, 3), keepdim=True)
        contraste = 1 + (torch.rand(b, 1, 1, 1, device=dispositivo, generator=generador) - 0.5) * 0.4
        x = ((x - media) * contraste + media) * brillo
        x = x.clamp_(0, 1)
    media_in = torch.tensor(MEDIA_IMAGENET, device=dispositivo).view(1, 3, 1, 1)
    desv_in = torch.tensor(DESV_IMAGENET, device=dispositivo).view(1, 3, 1, 1)
    return (x - media_in) / desv_in


def predecir(modelo, ruta_x, indices, dispositivo, batch: int = 512, workers: int = 2):
    import torch
    from torch.utils.data import DataLoader

    indices = np.asarray(indices)
    carga = DataLoader(
        _Recortes(ruta_x, np.zeros(int(indices.max()) + 1 if len(indices) else 0, int), indices), batch_size=batch, shuffle=False, num_workers=workers
    )
    modelo.eval()
    preds = []
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=dispositivo.type == "cuda"):
        for xb, _ in carga:
            preds.append(modelo(_a_tensor(xb, dispositivo, False)).argmax(1).cpu().numpy())
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    return (np.concatenate(preds) if preds else np.zeros(0, int)), time.perf_counter() - t0


def _limitar(n: int, maximo, semilla: int) -> np.ndarray:
    idx = np.arange(n)
    if maximo and n > maximo:
        idx = np.sort(np.random.default_rng(semilla).choice(n, size=maximo, replace=False))
    return idx


def entrenar(config: Path, datos: Path, salida: Path, rapido: bool = False, semilla=None) -> Path:
    import torch
    from torch.utils.data import DataLoader

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if semilla is not None:
        cfg["semilla"] = semilla
    if rapido:
        cfg = {**cfg, **RAPIDO, "nombre": cfg["nombre"] + "_rapido"}
    cfg["datos"] = str(datos)
    fijar_semillas(cfg["semilla"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dispositivo.type != "cuda":
        print("AVISO: no hay GPU; M1 entrenará en CPU (muy lento salvo con --rapido).")

    cache = Path(salida) / "cache_m1"
    t0 = time.perf_counter()
    c_tr, c_va = construir_cache(datos, "train", cache, cfg["lado"]), construir_cache(datos, "val", cache, cfg["lado"])
    t_cache = time.perf_counter() - t0
    _, y_tr, _ = cargar(c_tr)
    _, y_va, _ = cargar(c_va)
    idx_tr = _limitar(len(y_tr), cfg.get("max_train"), cfg["semilla"])
    idx_va = _limitar(len(y_va), cfg.get("max_val"), cfg["semilla"])
    print(f"Recortes: train {len(idx_tr)} / val {len(idx_va)} ({cfg['lado']} px; caché en {t_cache:.0f} s)")

    corrida = Path(salida) / id_corrida(cfg["nombre"], cfg["semilla"])
    corrida.mkdir(parents=True, exist_ok=False)
    (corrida / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    guardar_json(corrida / "entorno.json", capturar_entorno(Path(datos)))

    modelo, cabeza = construir_modelo(cfg["arquitectura"], len(NOMBRES), cfg.get("preentrenado", True))
    modelo.to(dispositivo)
    pesos = torch.tensor(pesos_por_clase(y_tr[idx_tr], len(NOMBRES)), dtype=torch.float32, device=dispositivo)
    perdida = torch.nn.CrossEntropyLoss(weight=pesos)
    gen_datos = torch.Generator().manual_seed(cfg["semilla"])
    carga = DataLoader(
        _Recortes(c_tr["X"], y_tr, idx_tr),
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=cfg.get("workers", 2),
        generator=gen_datos,
        drop_last=len(idx_tr) > cfg["batch"],
        pin_memory=dispositivo.type == "cuda",
    )
    gen_aum = torch.Generator(device=dispositivo).manual_seed(cfg["semilla"])
    escalador = torch.amp.GradScaler(dispositivo.type, enabled=dispositivo.type == "cuda" and cfg.get("amp", True))

    def _epoca(optimizador, planificador=None):
        modelo.train()
        total, n = 0.0, 0
        for xb, yb in carga:
            yb = yb.to(dispositivo, non_blocking=True)
            x = _a_tensor(xb, dispositivo, True, gen_aum)
            optimizador.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=escalador.is_enabled()):
                loss = perdida(modelo(x), yb)
            escalador.scale(loss).backward()
            escalador.step(optimizador)
            escalador.update()
            if planificador is not None:
                planificador.step()
            total += float(loss) * len(yb)
            n += len(yb)
        return total / max(n, 1)

    historial, mejor_f1, mejor_epoca = [], -1.0, None
    ruta_pesos = corrida / "modelo_m1.pt"
    t_train = time.perf_counter()

    def _registrar(etapa, epoca, loss):
        nonlocal mejor_f1, mejor_epoca
        pred, _ = predecir(modelo, c_va["X"], idx_va, dispositivo, workers=cfg.get("workers", 2))
        met = M.metricas_clasificacion(y_va[idx_va], pred, len(NOMBRES))
        fila = {
            "etapa": etapa,
            "epoca": epoca,
            "perdida_train": round(loss, 5),
            "exactitud_val": round(met["exactitud"], 5),
            "f1_macro_val": round(met["f1_macro"], 5),
            "segundos": round(time.perf_counter() - t_train, 1),
        }
        historial.append(fila)
        print(fila)
        if met["f1_macro"] > mejor_f1:
            mejor_f1, mejor_epoca = met["f1_macro"], f"{etapa}-{epoca}"
            torch.save(modelo.state_dict(), ruta_pesos)

    # Etapa 1: solo la cabeza
    for p in modelo.parameters():
        p.requires_grad = False
    for p in cabeza.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(cabeza.parameters(), lr=cfg["lr_cabeza"], weight_decay=cfg["weight_decay"])
    for e in range(1, cfg["epocas_cabeza"] + 1):
        _registrar("cabeza", e, _epoca(opt))
    # Etapa 2: toda la red
    for p in modelo.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg["epocas"] * len(carga)))
    for e in range(1, cfg["epocas"] + 1):
        _registrar("completa", e, _epoca(opt, plan))
    t_train = time.perf_counter() - t_train

    with open(corrida / "historial.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(historial[0]))
        w.writeheader()
        w.writerows(historial)
    guardar_json(
        ruta_manifiesto(ruta_pesos),
        {
            "archivo": ruta_pesos.name,
            "sha256": sha256_archivo(ruta_pesos),
            "familia": "M1",
            "arquitectura": cfg["arquitectura"],
            "preentrenado_imagenet": cfg.get("preentrenado", True),
            "lado": cfg["lado"],
            "clases": NOMBRES,
            "mejor_epoca": mejor_epoca,
            "datos_entrenamiento": f"recortes de cajas anotadas, partición train ({len(idx_tr)}), dataset {version_dataset(datos)[:12]}",
            "validado_con_datos_reales": False,
            "corrida_id": corrida.name,
        },
    )
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=dispositivo, weights_only=True))
    pred, t_pred = predecir(modelo, c_va["X"], idx_va, dispositivo, workers=cfg.get("workers", 2))
    guardar_json(
        corrida / "tiempos.json",
        {
            "cache_recortes_s": t_cache,
            "entrenamiento_s": t_train,
            "gpu": dispositivo.type == "cuda",
            "ms_por_recorte_prediccion_val": 1000 * t_pred / max(1, len(idx_va)),
        },
    )
    met = R.metricas(y_va[idx_va], pred, t_pred, dispositivo.type)
    print(_guardar_reporte(corrida, "val", met, cfg, mejor_epoca))
    return corrida


def _guardar_reporte(corrida: Path, split: str, met: dict, cfg: dict, mejor_epoca) -> str:
    contexto = [
        f"Recortes de cajas anotadas: {met['recortes']} ({cfg['lado']}×{cfg['lado']} px)",
        f"{cfg['arquitectura']} {'preentrenada en ImageNet (transfer learning)' if cfg.get('preentrenado', True) else 'SIN preentrenar'}; "
        f"{cfg['epocas_cabeza']} épocas de cabeza + {cfg['epocas']} de ajuste completo; mejor época (F1 macro en val): {mejor_epoca}",
    ]
    titulo = f"Evaluación M1 ({cfg['arquitectura']}) — corrida `{corrida.name}` — partición `{split}`"
    return R.guardar(corrida, split, met, titulo, contexto, "python -m ml.m1_cnn.entrenar", "M1")


def evaluar_final(corrida: Path, forzar: bool = False) -> str:
    import torch

    registro = corrida / "evaluacion_final.json"
    if registro.is_file() and not forzar:
        raise SystemExit(f"Ya existe {registro}: el test ya se usó. Repetir exige --forzar (queda registrado).")
    cfg = yaml.safe_load((corrida / "config.yaml").read_text(encoding="utf-8"))
    ruta_pesos = corrida / "modelo_m1.pt"
    manifiesto, _ = verificar_archivo(ruta_pesos, ("sha256",))
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, _ = construir_modelo(cfg["arquitectura"], len(NOMBRES), preentrenado=False)
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=dispositivo, weights_only=True))
    modelo.to(dispositivo)
    c_te = construir_cache(Path(cfg["datos"]), "test", corrida.parent / "cache_m1", cfg["lado"])
    _, y_te, _ = cargar(c_te)
    pred, t_pred = predecir(modelo, c_te["X"], np.arange(len(y_te)), dispositivo, workers=cfg.get("workers", 2))
    met = R.metricas(y_te, pred, t_pred, dispositivo.type)
    resumen = _guardar_reporte(corrida, "test", met, cfg, manifiesto.get("mejor_epoca"))
    previo = json.loads(registro.read_text(encoding="utf-8")) if registro.is_file() else {"ejecuciones": []}
    previo["ejecuciones"].append({"forzado": bool(forzar), "exactitud": met["exactitud"], "f1_macro": met["f1_macro"]})
    guardar_json(registro, previo)
    return resumen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--datos", type=Path)
    p.add_argument("--salida", type=Path)
    p.add_argument("--semilla", type=int)
    p.add_argument("--rapido", action="store_true", help="prueba de humo: pocos recortes y 1+1 épocas")
    p.add_argument("--corrida", type=Path)
    p.add_argument("--final", action="store_true")
    p.add_argument("--forzar", action="store_true")
    a = p.parse_args(argv)
    if a.final:
        if not a.corrida:
            p.error("--final requiere --corrida")
        print(evaluar_final(a.corrida, a.forzar))
        return 0
    if not (a.config and a.datos and a.salida):
        p.error("--config, --datos y --salida son obligatorios")
    entrenar(a.config, a.datos, a.salida, a.rapido, a.semilla)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
