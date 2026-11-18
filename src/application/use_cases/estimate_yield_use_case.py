"""
Use Case: AnalizarImagenCosechaUseCase
"""
import io
import unicodedata
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ...domain.exceptions.domain_exceptions import (
    ImagenInvalidaException,
    ParametrosInvalidosException,
    UmbralInvalidoException,
)
from ...domain.models.agronomic_params import ParametrosAgronomicos
from ...domain.models.coffee_cherry import DeteccionFruto, EstadoMadurez
from ...domain.models.modelo_info import ModeloInfo
from ...domain.models.supuestos import CALIBRACION, CLAVES_PARAMETROS, DATO, LITERATURA, TODO_FUENTE, CatalogoSupuestos
from ...domain.ports.fruit_detector_port import FruitDetectorPort
from ...domain.services.yield_calculator import CalculadoraRendimientoService
from ..dtos.estimate_dto import RespuestaEstimacionDTO, SolicitudEstimacionDTO


class AnalizarImagenCosechaUseCase:
    def __init__(self, detector_adapter: FruitDetectorPort, catalogo: Optional[CatalogoSupuestos] = None):
        self.detector_adapter = detector_adapter
        # Sin catálogo: valores por defecto del dominio, SIN fuentes (todas las advertencias activas).
        self.catalogo = catalogo or CatalogoSupuestos.desde_valores({k: getattr(ParametrosAgronomicos(), k) for k in CLAVES_PARAMETROS})

    def ejecutar(self, solicitud: SolicitudEstimacionDTO) -> RespuestaEstimacionDTO:
        if not (0.0 <= solicitud.umbral_confianza <= 1.0):
            raise UmbralInvalidoException(f"Umbral invalido: {solicitud.umbral_confianza}")

        if not solicitud.imagen_bytes:
            raise ImagenInvalidaException("No se enviaron datos de imagen.")

        try:
            imagen = Image.open(io.BytesIO(solicitud.imagen_bytes))
            imagen = ImageOps.exif_transpose(imagen)  # respeta la rotacion de las fotos de celular
            imagen = imagen.convert("RGB")
        except Exception as e:
            raise ImagenInvalidaException(f"Error abriendo imagen: {e}") from e

        usados = {k: self.catalogo.factores[k].usado_con(getattr(solicitud, k)) for k in CLAVES_PARAMETROS}
        try:
            parametros = ParametrosAgronomicos(
                peso_medio_fruto_g=float(usados["peso_medio_fruto_g"].valor),
                factor_oclusion=float(usados["factor_oclusion"].valor),
                plantas_muestra=int(usados["plantas_muestra"].valor),
                plantas_totales_lote=int(usados["plantas_totales_lote"].valor),
            )
        except (TypeError, ValueError) as e:
            raise ParametrosInvalidosException(str(e)) from e
        pergamino = self.catalogo.factores.get("factor_cereza_pergamino")
        factor_pergamino = pergamino.valor if pergamino is not None and pergamino.valor is not None and pergamino.tiene_fuente else None
        supuestos = list(usados.values()) + ([pergamino] if pergamino is not None else [])

        info = self.detector_adapter.info()
        detecciones = self.detector_adapter.detectar_frutos(imagen, solicitud.umbral_confianza)
        estados_cosechables = [EstadoMadurez(e) for e in self.catalogo.estados_cosechables] or None
        reporte = CalculadoraRendimientoService.calcular(detecciones, parametros, factor_pergamino, estados_cosechables)
        imagen_anotada_bytes = self._generar_imagen_anotada(imagen, detecciones, info)

        conteo_str_dict = {
            (e.value if hasattr(e, "value") else str(e)): count
            for e, count in reporte.conteo_por_estado.items()
        }

        detalle_detecciones = [
            {
                "id": d.id_deteccion,
                "estado": (d.estado.value if hasattr(d.estado, "value") else str(d.estado)),
                "confianza": round(d.confianza, 3),
                "bbox": [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max],
            }
            for d in detecciones
        ]

        return RespuestaEstimacionDTO(
            total_frutos=reporte.total_frutos,
            conteo_por_estado=conteo_str_dict,
            rendimiento_estimado_kg=reporte.rendimiento_estimado_kg,
            porcentaje_madurez=reporte.porcentaje_madurez,
            imagen_anotada_bytes=imagen_anotada_bytes,
            detecciones_detalle=detalle_detecciones,
            modelo=info.como_dict(),
            advertencias=list(info.advertencias) + self._advertencias_rendimiento(supuestos),
            supuestos=[x.como_dict() for x in supuestos],
            unidad_rendimiento=self.catalogo.salida,
            rendimiento_pergamino_kg=reporte.rendimiento_pergamino_kg,
            estados_cosechables=list(self.catalogo.estados_cosechables),
            fuente_estados_cosechables=self.catalogo.fuente_estados_cosechables,
            frutos_cosechables=reporte.frutos_cosechables,
            rendimiento_cosechable_kg=reporte.rendimiento_cosechable_kg,
            rendimiento_cosechable_pergamino_kg=reporte.rendimiento_cosechable_pergamino_kg,
            version_parametros=self.catalogo.version_parametros,
            unidad_muestreo=self.catalogo.supuesto_muestreo or None,
        )

    def _advertencias_rendimiento(self, supuestos) -> List[str]:
        avisos = []
        sin_fuente = [x.descripcion.lower() for x in supuestos if x.valor is not None and not x.tiene_fuente and x.tipo == LITERATURA]
        if sin_fuente:
            avisos.append(f"RENDIMIENTO INDICATIVO: factores sin fuente documental ({TODO_FUENTE}): {', '.join(sin_fuente)}.")
        for x in supuestos:
            if x.tipo == CALIBRACION and x.valor is not None and not x.tiene_fuente:
                avisos.append(f"Sin calibración local: {x.descripcion.lower()} = {x.valor:g} ({x.estado or 'pendiente de calibración'}).")
        datos_ejemplo = [x.descripcion.lower() for x in supuestos if x.valor is not None and not x.tiene_fuente and x.tipo == DATO]
        if datos_ejemplo:
            avisos.append(f"Datos del lote con valor de ejemplo (ingrese los suyos): {', '.join(datos_ejemplo)}.")
        con_fuente = [x for x in supuestos if x.tipo == LITERATURA and x.tiene_fuente and x.valor is not None]
        if con_fuente and not self.catalogo.calibrado_localmente:
            version = f" (parámetros {self.catalogo.version_parametros})" if self.catalogo.version_parametros else ""
            avisos.append(f"ESTIMACIÓN INDICATIVA basada en parámetros bibliográficos, no calibrados localmente{version}.")
        avisos.extend(self.catalogo.simplificaciones)
        if not self.catalogo.muestreo_definido:
            avisos.append(f"Unidad de muestreo sin definir: {self.catalogo.supuesto_muestreo}")
        if not self.catalogo.validado_con_cosecha_real:
            avisos.append(f"El rendimiento ({self.catalogo.salida}) no está validado con cosecha real.")
        return avisos

    def _generar_imagen_anotada(self, imagen: Image.Image, detecciones: List[DeteccionFruto], info: ModeloInfo) -> bytes:
        """Dibuja las cajas. Si el modelo no está validado, añade arriba una franja con la
        advertencia (la imagen suele compartirse sin el resto del reporte)."""
        ancho, alto = imagen.size
        lado = max(ancho, alto)
        grosor = max(2, round(lado / 300))  # legible tanto en miniaturas como en fotos grandes
        fuente = _fuente(max(11, round(lado / 60)))
        etiqueta = info.etiqueta
        alto_franja = round(fuente.size * 1.8) if etiqueta else 0

        lienzo = Image.new("RGB", (ancho, alto + alto_franja), "#B03A2E")
        lienzo.paste(imagen, (0, alto_franja))
        draw = ImageDraw.Draw(lienzo)
        if etiqueta:
            margen = max(4, fuente.size // 2)
            texto, fuente_franja = _ajustar_texto(
                draw, [f"AVISO: {etiqueta} | {info.nombre}", f"AVISO: {etiqueta}"], fuente.size, ancho - 2 * margen
            )
            draw.text((margen, (alto_franja - fuente_franja.size) // 2), texto, fill="white", font=fuente_franja)

        colores = {
            EstadoMadurez.VERDE: "#2ECC71",
            EstadoMadurez.PINTON: "#F1C40F",
            EstadoMadurez.MADURO: "#E74C3C",
            EstadoMadurez.SOBREMADURO: "#8E44AD",
        }
        for det in detecciones:
            color = colores.get(det.estado, "#3498DB")
            y0 = det.bbox.y_min + alto_franja
            draw.rectangle([det.bbox.x_min, y0, det.bbox.x_max, det.bbox.y_max + alto_franja], outline=color, width=grosor)
            estado_str = det.estado.value if hasattr(det.estado, "value") else str(det.estado)
            draw.text(
                (det.bbox.x_min, max(alto_franja, y0 - fuente.size - 2)),
                f"{estado_str} {int(det.confianza * 100)}%",
                fill=color,
                font=fuente,
            )

        buf = io.BytesIO()
        lienzo.save(buf, format="JPEG", quality=92)
        return buf.getvalue()


def _fuente(tam: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=tam)  # fuente escalable incluida en Pillow >= 10.1


def _ascii(texto: str) -> str:
    """La fuente incluida en Pillow no tiene tildes ni guiones largos."""
    texto = texto.replace("–", "-").replace("—", "-")
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def _ajustar_texto(draw: ImageDraw.ImageDraw, candidatos: List[str], tam: int, ancho_max: float):
    """Primer candidato que cabe; si ninguno cabe, el más corto con la fuente reducida."""
    fuente = _fuente(tam)
    for texto in map(_ascii, candidatos):
        if draw.textlength(texto, font=fuente) <= ancho_max:
            return texto, fuente
    texto = _ascii(candidatos[-1])
    while tam > 8 and draw.textlength(texto, font=_fuente(tam)) > ancho_max:
        tam -= 1
    return texto, _fuente(tam)
