"""
Use Case: AnalizarImagenCosechaUseCase
"""
import io
from PIL import Image, ImageDraw
from typing import List, Dict, Any
from ...domain.ports.fruit_detector_port import FruitDetectorPort
from ...domain.models.agronomic_params import ParametrosAgronomicos
from ...domain.models.coffee_cherry import DeteccionFruto, EstadoMadurez
from ...domain.services.yield_calculator import CalculadoraRendimientoService
from ...domain.exceptions.domain_exceptions import UmbralInvalidoException, ImagenInvalidaException
from ..dtos.estimate_dto import SolicitudEstimacionDTO, RespuestaEstimacionDTO

class AnalizarImagenCosechaUseCase:
    def __init__(self, detector_adapter: FruitDetectorPort):
        self.detector_adapter = detector_adapter

    def ejecutar(self, solicitud: SolicitudEstimacionDTO) -> RespuestaEstimacionDTO:
        if not (0.0 <= solicitud.umbral_confianza <= 1.0):
            raise UmbralInvalidoException(f"Umbral invalido: {solicitud.umbral_confianza}")
        
        if not solicitud.imagen_bytes:
            raise ImagenInvalidaException("No se enviaron datos de imagen.")

        try:
            imagen = Image.open(io.BytesIO(solicitud.imagen_bytes)).convert("RGB")
        except Exception as e:
            raise ImagenInvalidaException(f"Error abriendo imagen: {str(e)}")

        parametros = ParametrosAgronomicos(
            peso_medio_fruto_g=solicitud.peso_medio_fruto_g,
            factor_oclusion=solicitud.factor_oclusion,
            plantas_muestra=solicitud.plantas_muestra,
            plantas_totales_lote=solicitud.plantas_totales_lote
        )

        detecciones = self.detector_adapter.detectar_frutos(imagen, solicitud.umbral_confianza)
        reporte = CalculadoraRendimientoService.calcular(detecciones, parametros)
        imagen_anotada_bytes = self._generar_imagen_anotada(imagen, detecciones)

        conteo_str_dict = {
            (e.value if hasattr(e, "value") else str(e)): count 
            for e, count in reporte.conteo_por_estado.items()
        }
        
        detalle_detecciones = [
            {
                "id": d.id_deteccion,
                "estado": (d.estado.value if hasattr(d.estado, "value") else str(d.estado)),
                "confianza": round(d.confianza, 3),
                "bbox": [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max]
            }
            for d in detecciones
        ]

        return RespuestaEstimacionDTO(
            total_frutos=reporte.total_frutos,
            conteo_por_estado=conteo_str_dict,
            rendimiento_estimado_kg=reporte.rendimiento_estimado_kg,
            porcentaje_madurez=reporte.porcentaje_madurez,
            imagen_anotada_bytes=imagen_anotada_bytes,
            detecciones_detalle=detalle_detecciones
        )

    def _generar_imagen_anotada(self, imagen: Image.Image, detecciones: List[DeteccionFruto]) -> bytes:
        img_copy = imagen.copy()
        draw = ImageDraw.Draw(img_copy)
        colores = {
            EstadoMadurez.VERDE: "#2ECC71",
            EstadoMadurez.PINTON: "#F1C40F",
            EstadoMadurez.MADURO: "#E74C3C",
            EstadoMadurez.SOBREMADURO: "#8E44AD"
        }

        for det in detecciones:
            color = colores.get(det.estado, "#3498DB")
            box = [det.bbox.x_min, det.bbox.y_min, det.bbox.x_max, det.bbox.y_max]
            draw.rectangle(box, outline=color, width=3)
            estado_str = det.estado.value if hasattr(det.estado, "value") else str(det.estado)
            draw.text((det.bbox.x_min, max(0, det.bbox.y_min - 12)), f"{estado_str} {int(det.confianza*100)}%", fill=color)

        buf = io.BytesIO()
        img_copy.save(buf, format="JPEG")
        return buf.getvalue()
