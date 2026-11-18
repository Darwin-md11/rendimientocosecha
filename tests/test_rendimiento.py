"""Módulo de rendimiento (versión mínima): fórmula con valores a mano, fuentes, TODO(fuente) y pergamino opcional."""

import io

import pytest
from PIL import Image

from src.application.dtos.estimate_dto import SolicitudEstimacionDTO
from src.application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from src.domain.exceptions.domain_exceptions import ParametrosInvalidosException
from src.domain.models.agronomic_params import ParametrosAgronomicos
from src.domain.models.coffee_cherry import BoundingBox, DeteccionFruto, EstadoMadurez
from src.domain.models.modelo_info import FamiliaModelo, ModeloInfo
from src.domain.models.supuestos import DATO, INGRESADO_POR_USUARIO, LITERATURA, TODO_FUENTE, Supuesto
from src.domain.ports.fruit_detector_port import FruitDetectorPort
from src.domain.services.yield_calculator import CalculadoraRendimientoService
from src.infrastructure.config.agronomia import RUTA_AGRONOMIA, cargar_catalogo


class DiezCerezas(FruitDetectorPort):
    def detectar_frutos(self, imagen, umbral_confianza=0.40):
        return [DeteccionFruto(f"D{i}", EstadoMadurez.MADURO, 0.9, BoundingBox(i, 0, i + 1, 1)) for i in range(10)]

    def info(self):
        return ModeloInfo("fijo", FamiliaModelo.M2_YOLO, "1", "prueba", True, dominio_validacion="x", validado_en_dominio_objetivo=True)


def _jpeg():
    b = io.BytesIO()
    Image.new("RGB", (20, 20)).save(b, format="JPEG")
    return b.getvalue()


def test_formula_a_mano():
    dets = DiezCerezas().detectar_frutos(None)
    # 10 cerezas × 1,25 × 1,8 g / 1000 × (2500 / 1) = 56,25 kg; pergamino con factor 0,2 = 11,25 kg
    r = CalculadoraRendimientoService.calcular(dets, ParametrosAgronomicos(1.8, 1.25, 1, 2500), factor_cereza_pergamino=0.2)
    assert r.rendimiento_estimado_kg == pytest.approx(56.25) and r.rendimiento_pergamino_kg == pytest.approx(11.25)
    assert CalculadoraRendimientoService.calcular(dets, ParametrosAgronomicos(1.8, 1.25, 4, 2500)).rendimiento_pergamino_kg is None
    assert CalculadoraRendimientoService.calcular(dets, ParametrosAgronomicos(2.0, 1.0, 4, 2000)).rendimiento_estimado_kg == pytest.approx(10.0)


def test_catalogo_del_repositorio_marca_todo_fuente():
    cat = cargar_catalogo()
    assert cat.salida == "kg de café cereza" and cat.origen == RUTA_AGRONOMIA.name
    assert cat.factores["peso_medio_fruto_g"].fuente_mostrada == TODO_FUENTE
    assert cat.factores["factor_cereza_pergamino"].valor is None  # sin factor con fuente: no se calcula pergamino


def test_respuesta_con_catalogo_del_repo_advierte_y_no_calcula_pergamino():
    r = AnalizarImagenCosechaUseCase(DiezCerezas(), cargar_catalogo()).ejecutar(SolicitudEstimacionDTO(imagen_bytes=_jpeg()))
    assert r.rendimiento_estimado_kg == pytest.approx(56.25) and r.rendimiento_pergamino_kg is None
    assert r.unidad_rendimiento == "kg de café cereza"
    textos = " ".join(r.advertencias)
    assert "RENDIMIENTO INDICATIVO" in textos and "peso medio de una cereza" in textos and "Unidad de muestreo sin definir" in textos
    assert {s["clave"]: s["fuente"] for s in r.supuestos}["factor_oclusion"] == TODO_FUENTE


def test_valores_del_usuario_y_su_fuente():
    cat = cargar_catalogo()
    sol = SolicitudEstimacionDTO(imagen_bytes=_jpeg(), peso_medio_fruto_g=2.0, plantas_totales_lote=3000, plantas_muestra=1)
    r = AnalizarImagenCosechaUseCase(DiezCerezas(), cat).ejecutar(sol)
    fuentes = {s["clave"]: s for s in r.supuestos}
    assert fuentes["plantas_totales_lote"]["fuente"] == INGRESADO_POR_USUARIO and fuentes["plantas_totales_lote"]["tiene_fuente"]
    # Un factor de literatura escrito por el usuario sigue sin fuente documental
    assert fuentes["peso_medio_fruto_g"]["valor"] == 2.0 and not fuentes["peso_medio_fruto_g"]["tiene_fuente"]
    assert r.rendimiento_estimado_kg == pytest.approx(10 * 1.25 * 0.002 * 3000)
    assert "Datos del lote con valor de ejemplo" not in " ".join(r.advertencias)


def test_supuesto_de_literatura_igual_al_documentado_conserva_la_fuente():
    s = Supuesto("peso_medio_fruto_g", "Peso", LITERATURA, 1.8, "g", "Cenicafé (prueba)")
    assert s.usado_con(1.8).fuente == "Cenicafé (prueba)" and s.usado_con(1.9).fuente == INGRESADO_POR_USUARIO
    assert Supuesto("plantas_totales_lote", "Plantas", DATO, 10, "plantas").usado_con(10).tiene_fuente


def test_pergamino_solo_con_factor_y_fuente(tmp_path):
    texto = RUTA_AGRONOMIA.read_text(encoding="utf-8").replace(
        "    valor: null             # sin valor ni fuente: el reporte NO calcula pergamino\n    unidad: kg pergamino / kg cereza\n    fuente: null",
        "    valor: 0.2\n    unidad: kg pergamino / kg cereza\n    fuente: Fuente de prueba",
    )
    ruta = tmp_path / "agronomia.yaml"
    ruta.write_text(texto, encoding="utf-8")
    r = AnalizarImagenCosechaUseCase(DiezCerezas(), cargar_catalogo(ruta)).ejecutar(SolicitudEstimacionDTO(imagen_bytes=_jpeg()))
    assert r.rendimiento_pergamino_kg == pytest.approx(11.25)


def test_configuracion_invalida(tmp_path):
    malo = tmp_path / "a.yaml"
    malo.write_text(RUTA_AGRONOMIA.read_text(encoding="utf-8").replace("    valor: 1.8", "    valor: -1"), encoding="utf-8")
    with pytest.raises(ParametrosInvalidosException, match="> 0"):
        cargar_catalogo(malo)
    malo.write_text(
        RUTA_AGRONOMIA.read_text(encoding="utf-8").replace("    tipo: literatura\n    valor: 1.8", "    tipo: otro\n    valor: 1.8"), encoding="utf-8"
    )
    with pytest.raises(ParametrosInvalidosException, match="tipo"):
        cargar_catalogo(malo)
