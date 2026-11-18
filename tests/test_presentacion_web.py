"""Fragmentos HTML de la interfaz web (sin Gradio): honestidad, formato y seguridad."""
import re

from src.application.dtos.estimate_dto import RespuestaEstimacionDTO
from src.infrastructure.adapters.ui import presentacion as P


def _s(clave, descripcion, valor, unidad, fuente, tiene_fuente, estado=None):
    return dict(clave=clave, descripcion=descripcion, valor=valor, unidad=unidad, fuente=fuente, tiene_fuente=tiene_fuente, estado=estado)


def _respuesta(**cambios) -> RespuestaEstimacionDTO:
    base = dict(
        total_frutos=10, conteo_por_estado={"VERDE": 2, "PINTON": 0, "MADURO": 6, "SOBREMADURO": 2}, rendimiento_estimado_kg=46.25,
        porcentaje_madurez=60.0, imagen_anotada_bytes=b"", detecciones_detalle=[],
        modelo={"validado_con_datos_reales": True, "dominio_validacion": "Croppie test"},
        advertencias=["VALIDADO SOLO EN OTRO DOMINIO: <b>Croppie</b>", "Sin calibración local"],
        supuestos=[
            _s("peso_medio_fruto_g", "Peso medio", 1.85, "g", "Montilla-Pérez", True),
            _s("factor_oclusion", "Oclusión", 1.0, "", "TODO(fuente)", False, "pendiente"),
            _s("plantas_muestra", "Plantas en la foto", 1, "plantas", "decisión", True),
            _s("plantas_totales_lote", "Plantas del lote", 2500, "plantas", "usuario", True),
        ],
        rendimiento_pergamino_kg=9.25, estados_cosechables=["MADURO", "SOBREMADURO"], frutos_cosechables=8,
        rendimiento_cosechable_kg=37.0, rendimiento_cosechable_pergamino_kg=7.4, unidad_muestreo="Cada foto muestra una planta completa.",
    )
    base.update(cambios)
    return RespuestaEstimacionDTO(**base)


def test_numero_con_coma_decimal_y_miles():
    assert P.numero(46.25) == "46,25" and P.numero(2500, 0) == "2 500" and P.numero(None) == "—"


def test_tarjetas_y_calculo_con_los_numeros_del_resultado():
    r = _respuesta()
    t = P.tarjetas_html(r)
    assert "46,25" in t and "≈ 9,25 kg de pergamino seco" in t and ">80<" in t  # 8 de 10 cerezas cosechables
    c = P.calculo_html(r)
    assert "10 detectadas × 1,00 (oclusión) = 10,0" in c and "10,0 × 1,85 g ÷ 1000" in c and "= 0,0185 kg" in c
    assert "× 2\u202f500 plantas = <b>46,25 kg" in c and ">1 planta<" in c and ">1<" in c  # sin "1 plantas" ni "adimensional"
    assert "sin fuente" in c and "una planta completa" in c


def test_avisos_visibles_y_nunca_ok_si_hay_advertencias():
    html = P.avisos_html(_respuesta())
    assert P.TITULAR_AVISOS in html and "otra región" in html and "Ver las 2 advertencias" in html and "✅" not in html
    assert "&lt;b&gt;Croppie&lt;/b&gt;" in html and "<b>Croppie</b>" not in html  # texto escapado
    assert "✅" in P.avisos_html(_respuesta(advertencias=[]))


def test_madurez_en_orden_y_sin_cerezas():
    html = P.madurez_html(_respuesta())
    assert [m for m in re.findall(r"var\(--sk-(\w+)\)", html)][:3] == ["verde", "maduro", "sobremaduro"]  # sin segmento vacío
    assert "Pintón <b>0</b>" in html
    vacio = P.madurez_html(_respuesta(total_frutos=0, conteo_por_estado={}))
    assert "No se detectaron cerezas" in vacio


def test_paleta_definida_en_claro_y_oscuro():
    for modo in (":root", ".dark"):
        bloque = P.CSS.split(modo + " {", 1)[1].split("}", 1)[0]
        for estado in ("verde", "pinton", "maduro", "sobremaduro"):
            assert f"--sk-{estado}:" in bloque, (modo, estado)


def test_umbral_por_defecto_es_el_de_la_evaluacion():
    assert P.UMBRAL_POR_DEFECTO == 0.25
