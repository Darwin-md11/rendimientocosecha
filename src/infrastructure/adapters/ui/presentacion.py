"""
Textos de presentación compartidos por la UI (Gradio) y la CLI.

Viven aquí, sin depender de Gradio, para poder probar que ningún resultado de
un modelo no validado se muestra como "✅" o sin su advertencia.
"""
from typing import Any, Dict, List, Optional

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


# ============================================================================ Interfaz web (HTML)
# Fragmentos HTML para Gradio, sin depender de Gradio. Todo texto que viene de datos se escapa.
# Colores de madurez validados (dataviz/validate_palette.js) para daltonismo y contraste, en claro y oscuro;
# son la misma familia de tono que las cajas de la imagen anotada. Cada segmento lleva además su nombre y número.

TITULAR_AVISOS = "Resultado indicativo: revise las advertencias antes de usar el conteo o el rendimiento."
UMBRAL_POR_DEFECTO = 0.25  # el de la evaluación de M2 (conf_reporte): con él se midieron el R² y el MAE del conteo

ESTADOS_UI = (
    ("VERDE", "Verde", "inmaduro"),
    ("PINTON", "Pintón", "en transición"),
    ("MADURO", "Maduro", "listo para cosechar"),
    ("SOBREMADURO", "Sobremaduro", "pasado de cosecha"),
)

CSS = """
:root { --sk-verde:#227702; --sk-pinton:#bd821a; --sk-maduro:#9b1e22; --sk-sobremaduro:#7d51d2;
        --sk-aviso-fondo:#fff7e6; --sk-aviso-borde:#d08c00; --sk-aviso-texto:#5c3d00; }
.dark { --sk-verde:#297d0f; --sk-pinton:#b68b16; --sk-maduro:#bd413f; --sk-sobremaduro:#855bdc;
        --sk-aviso-fondo:#3a2c0f; --sk-aviso-borde:#d08c00; --sk-aviso-texto:#f3dfb4; }
.sk-encabezado { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px; padding:4px 2px 8px; }
.sk-titulo { font-size:1.5rem; font-weight:700; line-height:1.2; }
.sk-subtitulo { color:var(--body-text-color-subdued); font-size:.9rem; }
.sk-chip { display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:999px; font-size:.8rem; font-weight:600;
           border:1px solid var(--sk-aviso-borde); background:var(--sk-aviso-fondo); color:var(--sk-aviso-texto); }
.sk-chip.ok { border-color:#2e7d32; background:transparent; color:var(--body-text-color); }
.sk-tarjetas { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; }
.sk-grupo { font-size:.8rem; font-weight:600; color:var(--body-text-color-subdued); margin:14px 0 6px; }
.sk-grupo:first-child { margin-top:0; }
.sk-tarjeta { border:1px solid var(--border-color-primary); border-radius:12px; padding:14px 16px; background:var(--block-background-fill); }
.sk-tarjeta .et { font-size:.78rem; color:var(--body-text-color-subdued); text-transform:uppercase; letter-spacing:.04em; }
.sk-tarjeta .va { font-size:1.75rem; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums; }
.sk-tarjeta .un { font-size:.95rem; font-weight:500; color:var(--body-text-color-subdued); margin-left:4px; }
.sk-tarjeta .de { font-size:.8rem; color:var(--body-text-color-subdued); margin-top:4px; }
.sk-seccion { font-weight:600; margin:18px 0 8px; }
.sk-barra { display:flex; gap:2px; height:22px; border-radius:6px; overflow:hidden; background:var(--border-color-primary); }
.sk-barra div { min-width:3px; }
.sk-leyenda { display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:8px; font-size:.85rem; }
.sk-leyenda span.m { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:baseline; }
.sk-leyenda b { font-variant-numeric:tabular-nums; }
.sk-aviso { border:1px solid var(--sk-aviso-borde); border-left-width:5px; border-radius:10px; padding:12px 14px; background:var(--sk-aviso-fondo);
            color:var(--sk-aviso-texto); font-size:.88rem; margin-top:16px; }
.sk-aviso .ti { font-weight:700; }
.sk-aviso ul { margin:6px 0 0 18px; padding:0; } .sk-aviso li { margin:3px 0; }
.sk-aviso summary { cursor:pointer; font-weight:600; margin-top:8px; }
.sk-error { border:1px solid #c62828; border-left-width:5px; border-radius:10px; padding:12px 14px; }
.sk-vacio { border:1px dashed var(--border-color-primary); border-radius:12px; padding:28px; text-align:center;
            color:var(--body-text-color-subdued); }
.sk-tabla { width:100%; border-collapse:collapse; font-size:.86rem; border:none !important; box-shadow:none !important; table-layout:fixed; }
.sk-tabla tr { border:none !important; }
.sk-tabla th, .sk-tabla td { text-align:left; padding:8px 10px; vertical-align:top; background:transparent !important;
                             border:none !important; border-bottom:1px solid var(--border-color-primary) !important; }
.sk-tabla th { color:var(--body-text-color-subdued); font-weight:600; }
.sk-tabla td.n { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.sk-pasos { display:grid; grid-template-columns:max-content 1fr; gap:6px 14px; padding:12px 14px; border-radius:10px;
            background:var(--background-fill-secondary); font-size:.88rem; margin:6px 0 10px; }
.sk-pasos .p { color:var(--body-text-color-subdued); }
.sk-pasos .r { font-variant-numeric:tabular-nums; }
.sk-pendiente { color:var(--sk-aviso-texto); background:var(--sk-aviso-fondo); border-radius:6px; padding:1px 6px; font-size:.8rem; }
"""


def _e(texto) -> str:
    import html

    return html.escape(str(texto), quote=True)


def numero(valor, decimales: int = 2) -> str:
    """Formato colombiano: coma decimal y espacio fino como separador de miles (46,25 · 2 500)."""
    if valor is None:
        return "—"
    s = f"{valor:,.{decimales}f}".replace(",", " ").replace(".", ",")
    return s


def encabezado_html(info: ModeloInfo) -> str:
    chip = (f'<span class="sk-chip">⚠ {_e(info.etiqueta)}</span>' if info.etiqueta
            else '<span class="sk-chip ok">Modelo validado en la región</span>')
    return (
        '<div class="sk-encabezado"><div>'
        '<div class="sk-titulo">SIKAFE · Estimación de cosecha de café</div>'
        '<div class="sk-subtitulo">Conteo y madurez de cerezas con visión artificial, y rendimiento estimado del lote</div>'
        f'</div>{chip}</div>'
    )


def procedencia_html(info: ModeloInfo) -> str:
    filas = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in _filas_procedencia(info.como_dict()))
    avisos = "".join(f"<li>{_e(a)}</li>" for a in info.advertencias)
    aviso = f'<div class="sk-aviso"><div class="ti">⚠ {_e(info.etiqueta)}</div><ul>{avisos}</ul></div>' if info.etiqueta else ""
    return f'<table class="sk-tabla">{filas}</table>{aviso}'


def vacio_html() -> str:
    return ('<div class="sk-vacio">Suba la foto de <b>una planta completa</b> y pulse <b>Analizar</b>.<br>'
            'Aquí verá el conteo de cerezas, su madurez y el rendimiento estimado del lote.</div>')


def error_html(mensaje: str) -> str:
    return f'<div class="sk-error"><b>No se pudo analizar la foto.</b><br>{_e(mensaje)}</div>'


def _tarjeta(etiqueta: str, valor: str, unidad: str = "", detalle: str = "") -> str:
    un = f'<span class="un">{_e(unidad)}</span>' if unidad else ""
    de = f'<div class="de">{detalle}</div>' if detalle else ""
    return f'<div class="sk-tarjeta"><div class="et">{_e(etiqueta)}</div><div class="va">{_e(valor)}{un}</div>{de}</div>'


def porcentaje_cosechable(r: RespuestaEstimacionDTO) -> Optional[float]:
    if r.frutos_cosechables is None or not r.total_frutos:
        return None if r.frutos_cosechables is None else 0.0
    return 100 * r.frutos_cosechables / r.total_frutos


def _pergamino(v) -> str:
    return f"≈ {numero(v)} kg de pergamino seco" if v is not None else "pergamino: sin factor con fuente"


def tarjetas_html(r: RespuestaEstimacionDTO) -> str:
    pct = porcentaje_cosechable(r)
    estados = " + ".join(e.lower() for e in r.estados_cosechables) or "—"
    lote = _valor_supuesto(r, "plantas_totales_lote")
    planta = "".join([
        _tarjeta("Cerezas detectadas", numero(r.total_frutos, 0), "", "visibles en la foto"),
        _tarjeta("Listas para cosechar", numero(pct, 0) if pct is not None else "—", "%",
                 f"{_e(estados)}: {numero(r.frutos_cosechables or 0, 0)} cerezas"),
    ])
    lote_html = "".join([
        _tarjeta("Rendimiento potencial", numero(r.rendimiento_estimado_kg), "kg", f"café cereza · {_pergamino(r.rendimiento_pergamino_kg)}"),
        _tarjeta("Listo para cosechar", numero(r.rendimiento_cosechable_kg), "kg",
                 f"café cereza · {_pergamino(r.rendimiento_cosechable_pergamino_kg)}"),
    ])
    titulo_lote = f"Estimado para el lote · {numero(lote, 0)} plantas" if lote is not None else "Estimado para el lote"
    return (f'<div class="sk-grupo">En la planta fotografiada</div><div class="sk-tarjetas">{planta}</div>'
            f'<div class="sk-grupo">{_e(titulo_lote)}</div><div class="sk-tarjetas">{lote_html}</div>')


def filas_madurez(r: RespuestaEstimacionDTO) -> List[tuple]:
    """(clave, nombre, descripción, cerezas, % del total) por estado, en orden de maduración."""
    total = r.total_frutos or 0
    return [(k, n, d, r.conteo_por_estado.get(k, 0), (100 * r.conteo_por_estado.get(k, 0) / total) if total else 0.0)
            for k, n, d in ESTADOS_UI]


def madurez_html(r: RespuestaEstimacionDTO) -> str:
    filas = filas_madurez(r)
    titulo = '<div class="sk-seccion">Madurez de las cerezas detectadas</div>'
    if not r.total_frutos:
        return titulo + '<div class="sk-subtitulo">No se detectaron cerezas con el umbral elegido.</div>'
    segmentos = "".join(
        f'<div style="flex:{c} 1 0; background:var(--sk-{k.lower()})" title="{_e(n)}: {c} cerezas ({numero(p, 1)} %)"></div>'
        for k, n, _, c, p in filas if c > 0)
    leyenda = "".join(
        f'<div><span class="m" style="background:var(--sk-{k.lower()})"></span>{_e(n)} <b>{numero(c, 0)}</b> · {numero(p, 0)} %</div>'
        for k, n, _, c, p in filas)
    return f'{titulo}<div class="sk-barra" role="img" aria-label="Distribución por estado de madurez">{segmentos}</div>' \
           f'<div class="sk-leyenda">{leyenda}</div>'


def avisos_html(r: RespuestaEstimacionDTO) -> str:
    """Advertencias siempre visibles: un titular y la lista completa desplegable. Nunca un 'todo bien' si hay avisos."""
    if not r.advertencias:
        return f'<div class="sk-aviso" style="border-color:#2e7d32">{_e(ESTADO_OK)}</div>'
    modelo = (r.modelo or {}).get("dominio_validacion")
    clave = []
    if (r.modelo or {}).get("validado_con_datos_reales") and modelo:
        clave.append("El modelo se probó con fotos de otra región (Croppie, Uganda), no de Norte de Santander.")
    elif not (r.modelo or {}).get("validado_con_datos_reales"):
        clave.append("El modelo NO está validado con fotos reales.")
    clave.append("El rendimiento usa parámetros bibliográficos sin calibrar localmente, y no corrige las cerezas ocultas.")
    lista = "".join(f"<li>{_e(a)}</li>" for a in r.advertencias)
    return (f'<div class="sk-aviso"><div class="ti">⚠ {_e(TITULAR_AVISOS)}</div>'
            f'<ul>{"".join(f"<li>{_e(c)}</li>" for c in clave)}</ul>'
            f'<details><summary>Ver las {len(r.advertencias)} advertencias completas</summary><ul>{lista}</ul></details></div>')


def _valor_supuesto(r: RespuestaEstimacionDTO, clave: str):
    for s in r.supuestos:
        if s.get("clave") == clave:
            return s.get("valor")
    return None


def _valor_con_unidad(s: dict) -> str:
    v, unidad = s["valor"], (s.get("unidad") or "").strip()
    if v is None:
        return "—"
    texto = numero(v, 2 if v % 1 else 0)
    if unidad in ("", "adimensional"):
        return texto
    if v == 1 and unidad == "plantas":
        unidad = "planta"
    return f"{texto} {unidad}"


def calculo_html(r: RespuestaEstimacionDTO) -> str:
    """Cómo se calculó, paso a paso con los números de este resultado, y la tabla de supuestos con su fuente."""
    oc, peso = _valor_supuesto(r, "factor_oclusion"), _valor_supuesto(r, "peso_medio_fruto_g")
    muestra, lote = _valor_supuesto(r, "plantas_muestra"), _valor_supuesto(r, "plantas_totales_lote")
    factor_perg = _valor_supuesto(r, "factor_cereza_pergamino")
    partes = []
    if None not in (oc, peso, muestra, lote) and muestra:
        cerezas = r.total_frutos * oc
        kg_planta = cerezas * peso / 1000 / muestra
        kg, perg = numero(r.rendimiento_estimado_kg), numero(r.rendimiento_pergamino_kg)
        pasos = [
            ("Cerezas en la planta", f"{numero(r.total_frutos, 0)} detectadas × {numero(oc, 2)} (oclusión) = {numero(cerezas, 1)}"),
            ("Kilos por planta",
             f"{numero(cerezas, 1)} × {numero(peso, 2)} g ÷ 1000 ÷ {numero(muestra, 0)} planta(s) en la foto = {numero(kg_planta, 4)} kg"),
            ("Kilos en el lote", f"{numero(kg_planta, 4)} kg × {numero(lote, 0)} plantas = <b>{kg} kg de café cereza</b>"),
        ]
        if r.rendimiento_pergamino_kg is not None and factor_perg is not None:
            pasos.append(("Pergamino seco", f"{kg} × {numero(factor_perg, 2)} = {perg} kg"))
        if r.rendimiento_cosechable_kg is not None:
            estados = " y ".join(e.lower() for e in r.estados_cosechables)
            pasos.append(("Listo para cosechar", f"lo mismo con las {numero(r.frutos_cosechables or 0, 0)} cerezas {_e(estados)} = "
                          f"{numero(r.rendimiento_cosechable_kg)} kg"))
        filas = "".join(f'<div class="p">{_e(a)}</div><div class="r">{b}</div>' for a, b in pasos)
        partes.append(f'<div class="sk-seccion">Cálculo de este resultado</div><div class="sk-pasos">{filas}</div>')
        if r.fuente_estados_cosechables:
            partes.append(f'<div class="sk-subtitulo">Criterio de cosecha: {_e(r.fuente_estados_cosechables)}.</div>')
    if r.unidad_muestreo:
        partes.append(f'<div class="sk-seccion">Qué representa la foto</div><div>{_e(r.unidad_muestreo)}</div>')
    version = f" (versión {_e(r.version_parametros)})" if r.version_parametros else ""
    filas = []
    for s in r.supuestos:
        pend = "" if s.get("tiene_fuente") else ' <span class="sk-pendiente">sin fuente</span>'
        filas.append(f"<tr><td>{_e(s['descripcion'])}</td><td>{_e(_valor_con_unidad(s))}</td><td>{_e(s['fuente'])}{pend}</td>"
                     f"<td>{_e(s.get('estado') or '—')}</td></tr>")
    cols = '<colgroup><col style="width:30%"><col style="width:16%"><col style="width:32%"><col style="width:22%"></colgroup>'
    partes.append(f'<div class="sk-seccion">Supuestos del cálculo{version}</div><table class="sk-tabla">{cols}'
                  "<tr><th>Factor</th><th>Valor</th><th>Fuente</th><th>Estado</th></tr>" + "".join(filas) + "</table>")
    return "".join(partes)
