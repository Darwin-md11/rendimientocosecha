"""
Módulo de rendimiento: fórmula con valores a mano (incluido el ejemplo de la investigación del equipo), parámetros de
Cenicafé con su fuente, oclusión sin calibrar, "listo para cosechar" y advertencias.
"""

import io

import pytest
from PIL import Image

from src.application.dtos.estimate_dto import SolicitudEstimacionDTO
from src.application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from src.domain.exceptions.domain_exceptions import ParametrosInvalidosException
from src.domain.models.agronomic_params import ParametrosAgronomicos
from src.domain.models.coffee_cherry import BoundingBox, DeteccionFruto, EstadoMadurez
from src.domain.models.modelo_info import FamiliaModelo, ModeloInfo
from src.domain.models.supuestos import CALIBRACION, DATO, INGRESADO_POR_USUARIO, LITERATURA, TODO_FUENTE, Supuesto
from src.domain.ports.fruit_detector_port import FruitDetectorPort
from src.domain.services.yield_calculator import CalculadoraRendimientoService
from src.infrastructure.adapters.ui.presentacion import lineas_supuestos
from src.infrastructure.config.agronomia import RUTA_AGRONOMIA, cargar_catalogo

E = EstadoMadurez


def _dets(estados):
    return [DeteccionFruto(f"D{i}", e, 0.9, BoundingBox(i, 0, i + 1, 1)) for i, e in enumerate(estados)]


class Cerezas(FruitDetectorPort):
    """Detector fijo: 6 maduras, 2 sobremaduras, 2 verdes."""

    def detectar_frutos(self, imagen, umbral_confianza=0.40):
        return _dets([E.MADURO] * 6 + [E.SOBREMADURO] * 2 + [E.VERDE] * 2)

    def info(self):
        return ModeloInfo("fijo", FamiliaModelo.M2_YOLO, "1", "prueba", True, dominio_validacion="x", validado_en_dominio_objetivo=True)


def _jpeg():
    b = io.BytesIO()
    Image.new("RGB", (20, 20)).save(b, format="JPEG")
    return b.getvalue()


def _ejecutar(catalogo=None, **kw):
    return AnalizarImagenCosechaUseCase(Cerezas(), catalogo or cargar_catalogo()).ejecutar(SolicitudEstimacionDTO(imagen_bytes=_jpeg(), **kw))


# ------------------------------------------------------------------ fórmula
def test_ejemplo_de_la_investigacion_del_equipo():
    # docs/investigacion/parametros_agronomicos_cenicafe.md §5.3: 10 000 cerezas × 1,85 g = 18,5 kg de cereza; × 0,20 = 3,70 kg de pergamino
    r = CalculadoraRendimientoService.calcular(_dets([E.MADURO] * 10_000), ParametrosAgronomicos(1.85, 1.0, 1, 1), factor_cereza_pergamino=0.20)
    assert r.rendimiento_estimado_kg == pytest.approx(18.5) and r.rendimiento_pergamino_kg == pytest.approx(3.70)


def test_formula_con_oclusion_escala_y_listo_para_cosechar():
    dets = _dets([E.MADURO] * 6 + [E.SOBREMADURO] * 2 + [E.VERDE] * 2)
    # Total: 10 × 1,25 × 0,0018 × (2500 / 1) = 56,25 kg; cosechable (maduro + sobremaduro = 8): 45,0 kg; pergamino ×0,2
    r = CalculadoraRendimientoService.calcular(dets, ParametrosAgronomicos(1.8, 1.25, 1, 2500), 0.2, [E.MADURO, E.SOBREMADURO])
    assert r.rendimiento_estimado_kg == pytest.approx(56.25) and r.rendimiento_pergamino_kg == pytest.approx(11.25)
    assert (
        r.frutos_cosechables == 8
        and r.rendimiento_cosechable_kg == pytest.approx(45.0)
        and r.rendimiento_cosechable_pergamino_kg == pytest.approx(9.0)
    )
    sin = CalculadoraRendimientoService.calcular(dets, ParametrosAgronomicos(1.8, 1.25, 4, 2500))
    assert sin.rendimiento_pergamino_kg is None and sin.rendimiento_cosechable_kg is None and sin.frutos_cosechables is None


# ------------------------------------------------------------------ catálogo del repositorio (Cenicafé)
def test_catalogo_del_repositorio_con_parametros_de_cenicafe():
    cat = cargar_catalogo()
    peso, pergamino, oclusion = (cat.factores[k] for k in ("peso_medio_fruto_g", "factor_cereza_pergamino", "factor_oclusion"))
    assert peso.valor == 1.85 and peso.tiene_fuente and "Montilla-Pérez et al. (2008)" in peso.fuente and "Tabla 17" in peso.fuente
    assert pergamino.valor == 0.20 and pergamino.tiene_fuente and "Tabla 20" in pergamino.fuente
    assert oclusion.tipo == CALIBRACION and oclusion.valor == 1.0 and not oclusion.tiene_fuente and oclusion.fuente_mostrada == TODO_FUENTE
    assert "pendiente de calibración" in peso.estado and cat.version_parametros == "2026-11-18"
    assert cat.estados_cosechables == ("MADURO", "SOBREMADURO") and "Cartilla 19" in cat.fuente_estados_cosechables
    assert len(cat.simplificaciones) == 2 and not cat.calibrado_localmente


def test_respuesta_con_el_catalogo_del_repositorio():
    r = _ejecutar()
    # 10 × 1,0 × 0,00185 × 2500 = 46,25 kg de cereza; × 0,20 = 9,25 kg de pergamino; cosechables 8: 37,0 y 7,4
    assert r.rendimiento_estimado_kg == pytest.approx(46.25) and r.rendimiento_pergamino_kg == pytest.approx(9.25)
    assert (
        r.frutos_cosechables == 8
        and r.rendimiento_cosechable_kg == pytest.approx(37.0)
        and r.rendimiento_cosechable_pergamino_kg == pytest.approx(7.4)
    )
    textos = " ".join(r.advertencias)
    assert "RENDIMIENTO INDICATIVO" not in textos  # los factores bibliográficos ya tienen fuente
    for esperado in (
        "ESTIMACIÓN INDICATIVA",
        "no calibrados localmente",
        "Sin calibración local",
        "variedad Colombia",
        "Marín-López",
        "Unidad de muestreo",
    ):
        assert esperado in textos, esperado
    lineas = "\n".join(lineas_supuestos(r))
    assert "Listo para cosechar (maduro + sobremaduro: 8 cerezas): 37.0" in lineas and "Cartilla 19" in lineas and "versión 2026-11-18" in lineas


def test_valores_del_usuario_y_su_fuente():
    r = _ejecutar(peso_medio_fruto_g=2.0, plantas_totales_lote=3000, plantas_muestra=1)
    fuentes = {s["clave"]: s for s in r.supuestos}
    assert fuentes["plantas_totales_lote"]["fuente"] == INGRESADO_POR_USUARIO and fuentes["plantas_totales_lote"]["tiene_fuente"]
    # Un factor de literatura escrito por el usuario pierde la fuente documental
    assert fuentes["peso_medio_fruto_g"]["valor"] == 2.0 and not fuentes["peso_medio_fruto_g"]["tiene_fuente"]
    assert "RENDIMIENTO INDICATIVO" in " ".join(r.advertencias)
    assert r.rendimiento_estimado_kg == pytest.approx(10 * 1.0 * 0.002 * 3000)
    # El mismo valor documentado conserva la fuente
    assert {s["clave"]: s for s in _ejecutar(peso_medio_fruto_g=1.85).supuestos}["peso_medio_fruto_g"]["tiene_fuente"]


def test_reglas_de_fuente_por_tipo():
    lit = Supuesto("peso_medio_fruto_g", "Peso", LITERATURA, 1.85, "g", "Cenicafé (prueba)")
    assert lit.usado_con(1.85).fuente == "Cenicafé (prueba)" and lit.usado_con(1.9).fuente == INGRESADO_POR_USUARIO
    cal = Supuesto("factor_oclusion", "Oclusión", CALIBRACION, 1.0, "adimensional")
    assert not cal.tiene_fuente and not cal.usado_con(1.3).tiene_fuente  # una calibración no se "ingresa": se mide
    assert Supuesto("plantas_totales_lote", "Plantas", DATO, 10, "plantas").usado_con(10).tiene_fuente


def test_pergamino_desactivado_si_el_factor_no_tiene_fuente(tmp_path):
    texto = RUTA_AGRONOMIA.read_text(encoding="utf-8")
    viejo = '    fuente: "Montilla-Pérez et al. (2008), Cenicafé 59(2):120-142, Tabla 20"'
    assert viejo in texto
    ruta = tmp_path / "agronomia.yaml"
    ruta.write_text(texto.replace(viejo, "    fuente: null"), encoding="utf-8")
    r = _ejecutar(cargar_catalogo(ruta))
    assert r.rendimiento_pergamino_kg is None and r.rendimiento_cosechable_pergamino_kg is None


def test_configuracion_invalida(tmp_path):
    base = RUTA_AGRONOMIA.read_text(encoding="utf-8")
    malo = tmp_path / "a.yaml"
    for viejo, nuevo, mensaje in [
        ("    valor: 1.85", "    valor: -1", "> 0"),
        ("    tipo: calibracion", "    tipo: otro", "tipo"),
        ("estados_cosechables: [MADURO, SOBREMADURO]", "estados_cosechables: [MADURO, ROJO]", "estado cosechable"),
    ]:
        assert viejo in base, viejo
        malo.write_text(base.replace(viejo, nuevo, 1), encoding="utf-8")
        with pytest.raises(ParametrosInvalidosException, match=mensaje):
            cargar_catalogo(malo)
