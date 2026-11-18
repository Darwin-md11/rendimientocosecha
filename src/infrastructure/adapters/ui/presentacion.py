"""
Textos de presentación compartidos por la UI (Gradio) y la CLI.

Viven aquí, sin depender de Gradio, para poder probar que ningún resultado de
un modelo no validado se muestra como "✅" o sin su advertencia.
"""
from typing import Any, Dict, List

from ....application.dtos.estimate_dto import RespuestaEstimacionDTO
from ....domain.models.modelo_info import ModeloInfo

ESTADO_OK = "✅ Inferencia y estimación agronómica completadas (modelo validado con fotos de la región)."
ESTADO_NO_VALIDADO = (
    "⚠️ Inferencia completada con un modelo que NO está validado para este uso: "
    "revise las advertencias antes de usar los conteos o el rendimiento."
)


def lineas_supuestos(respuesta: RespuestaEstimacionDTO) -> List[str]:
    lineas = [f"Rendimiento estimado (potencial, todas las cerezas): {respuesta.rendimiento_estimado_kg} {respuesta.unidad_rendimiento}"]
    if respuesta.rendimiento_pergamino_kg is not None:
        lineas.append(f"  equivalente en café pergamino seco: {respuesta.rendimiento_pergamino_kg} kg")
    if respuesta.rendimiento_cosechable_kg is not None:
        estados = " + ".join(e.lower() for e in respuesta.estados_cosechables)
        lineas.append(f"Listo para cosechar ({estados}: {respuesta.frutos_cosechables} cerezas): "
                      f"{respuesta.rendimiento_cosechable_kg} {respuesta.unidad_rendimiento}")
        if respuesta.rendimiento_cosechable_pergamino_kg is not None:
            lineas.append(f"  equivalente en café pergamino seco: {respuesta.rendimiento_cosechable_pergamino_kg} kg")
        if respuesta.fuente_estados_cosechables:
            lineas.append(f"  criterio de cosecha: {respuesta.fuente_estados_cosechables}")
    version = f" (versión {respuesta.version_parametros})" if respuesta.version_parametros else ""
    lineas.append(f"Supuestos del cálculo{version}:")
    for s in respuesta.supuestos:
        valor = "—" if s["valor"] is None else f"{s['valor']:g} {s['unidad']}"
        estado = f"; {s['estado']}" if s.get("estado") else ""
        lineas.append(f"  - {s['descripcion']}: {valor} (fuente: {s['fuente']}{estado})")
    return lineas


def mensaje_estado(respuesta: RespuestaEstimacionDTO) -> str:
    supuestos = "\n\n" + "\n".join(lineas_supuestos(respuesta)) if respuesta.supuestos else ""
    if respuesta.advertencias:
        return ESTADO_NO_VALIDADO + "\n" + "\n".join(f"• {a}" for a in respuesta.advertencias) + supuestos
    return ESTADO_OK + supuestos


def _filas_procedencia(modelo: Dict[str, Any]) -> List[tuple]:
    sha = modelo.get("hash_pesos_sha256")
    return [
        ("Modelo", modelo["nombre"]),
        ("Familia", modelo["familia"]),
        ("Versión", modelo["version"]),
        ("Datos de entrenamiento", modelo["datos_entrenamiento"]),
        ("Validado con datos reales", "Sí" if modelo["validado_con_datos_reales"] else "NO"),
        ("Dominio de validación", modelo.get("dominio_validacion") or "—"),
        ("SHA-256 de los pesos", f"{sha[:16]}…" if sha else "—"),
        ("Fecha de entrenamiento", modelo.get("fecha_entrenamiento") or "no registrada"),
    ]


def banner_markdown(info: ModeloInfo) -> str:
    """Panel permanente de procedencia para la UI, con advertencia si no está validado."""
    tabla = "\n".join(
        ["| Procedencia del modelo | |", "|---|---|"]
        + [f"| {k} | {v} |" for k, v in _filas_procedencia(info.como_dict())]
    )
    if info.etiqueta:
        aviso = "\n".join(f"> {a}" for a in info.advertencias)
        return f"> ## ⚠️ {info.etiqueta}\n{aviso}\n\n{tabla}"
    return tabla


def lineas_cli(respuesta: RespuestaEstimacionDTO) -> List[str]:
    """Bloque de procedencia y advertencias para la CLI."""
    lineas = ["Procedencia del modelo:"] + [f"  {k}: {v}" for k, v in _filas_procedencia(respuesta.modelo)]
    if respuesta.supuestos:
        lineas += [""] + lineas_supuestos(respuesta)
    if respuesta.advertencias:
        lineas += ["", "!" * 58] + [f"! {a}" for a in respuesta.advertencias] + ["!" * 58]
    return lineas
